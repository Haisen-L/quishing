import os
import time
import json
import pathlib
import re
from html import unescape
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import Optional, List

import praw
import requests
from prawcore.exceptions import ResponseException, RequestException, Forbidden
from dotenv import load_dotenv
from pathlib import Path

# ------------------------------------------------------------
# Reddit Harvester
#   - Supports Script password flow or Refresh Token flow
#   - Harvests newest posts OR search results per subreddit (with optional time_filter for search)
#   - Writes JSON Lines, optional CSV
#   - Optional image downloader with subreddit, query, or subreddit-then-query folders
# ------------------------------------------------------------

# Load .env that sits next to this script, regardless of current working directory
load_dotenv(Path(__file__).with_name(".env"))

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "reddit-crawler by u_unknown")
USERNAME = os.getenv("REDDIT_USERNAME")
PASSWORD = os.getenv("REDDIT_PASSWORD")
REFRESH = os.getenv("REDDIT_REFRESH_TOKEN")

# Runtime knobs (can also be overridden via CLI env per run)
SUBREDDITS = [t.strip() for t in os.getenv("SUBREDDITS", "AskReddit").split(",") if t.strip()]
RAW_LIMIT = int(os.getenv("LIMIT", "250"))
LIMIT = None if RAW_LIMIT == 0 else RAW_LIMIT  # None = unbounded from API perspective
WRITE_CSV = os.getenv("WRITE_CSV", "0") == "1"

SAVE_MEDIA = os.getenv("SAVE_MEDIA", "0") == "1"
SAVE_NSFW = os.getenv("SAVE_NSFW", "0") == "1"
MEDIA_DIR = pathlib.Path(os.getenv("MEDIA_DIR", "data/media"))
MEDIA_SUBDIR_MODE = os.getenv("MEDIA_SUBDIR_MODE", "subreddit").lower()  # subreddit | query | subreddit_then_query
SEARCH_QUERY = os.getenv("SEARCH_QUERY") or None  # string -> use subreddit.search; None -> subreddit.new
SEARCH_QUERIES = os.getenv("SEARCH_QUERIES") or None  # semicolon or comma separated list of phrases
TIME_FILTER = (os.getenv("TIME_FILTER", "all") or "all").lower()  # for search: hour|day|week|month|year|all
COMBINE_NEW_AND_ALL_SEARCH = os.getenv("COMBINE_NEW_AND_ALL_SEARCH", "0") == "1"  # when 1 and SEARCH_QUERY/SEARCH_QUERIES is set: run new() for SUBREDDITS AND search() in r/all

# validate TIME_FILTER
if TIME_FILTER not in {"hour", "day", "week", "month", "year", "all"}:
    raise SystemExit("TIME_FILTER must be one of: hour, day, week, month, year, all")

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Missing client id or secret in env")

# Build Reddit instance for either auth mode
if REFRESH:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        refresh_token=REFRESH,
    )
else:
    if not (USERNAME and PASSWORD):
        raise SystemExit("Provide REDDIT_USERNAME and REDDIT_PASSWORD or a REDDIT_REFRESH_TOKEN")
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        username=USERNAME,
        password=PASSWORD,
    )

DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"

if SAVE_MEDIA:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_ids": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def submission_to_dict(s) -> dict:
    return {
        "id": s.id,
        "fullname": s.fullname if hasattr(s, "fullname") else f"t3_{s.id}",
        "created_utc": s.created_utc,
        "created_iso": datetime.fromtimestamp(s.created_utc, tz=timezone.utc).isoformat(),
        "subreddit": str(s.subreddit),
        "title": s.title,
        "author": str(s.author) if s.author else None,
        "score": int(s.score),
        "num_comments": int(s.num_comments),
        "url": s.url,
        "selftext": s.selftext,
        "over_18": bool(getattr(s, "over_18", False)),
        "link_flair_text": getattr(s, "link_flair_text", None),
        "is_self": bool(getattr(s, "is_self", False)),
        "is_gallery": bool(getattr(s, "is_gallery", False)),
        "post_hint": getattr(s, "post_hint", None),
        "domain": getattr(s, "domain", None),
        "permalink": f"https://www.reddit.com{s.permalink}"
    }


def is_image_url(u: str) -> bool:
    try:
        path = urlparse(u).path.lower()
        return any(path.endswith(ext) for ext in IMG_EXTS)
    except Exception:
        return False


def extract_image_urls(submission) -> List[str]:
    urls: List[str] = []

    # Galleries
    if getattr(submission, "is_gallery", False) and getattr(submission, "media_metadata", None):
        for item in submission.media_metadata.values():
            if not item:
                continue
            s = item.get("s") or {}
            u = s.get("u") or s.get("gif") or s.get("mp4")
            if u:
                u = unescape(u)
                if is_image_url(u):
                    urls.append(u)
        return dedup(urls)

    # Preview image
    try:
        if getattr(submission, "preview", None):
            u = submission.preview["images"][0]["source"]["url"]
            u = unescape(u)
            if is_image_url(u):
                urls.append(u)
    except Exception:
        pass

    # Direct link to an image
    if submission.url and is_image_url(submission.url):
        urls.append(submission.url)

    # i.redd.it without extension sometimes works by adding .jpg
    if submission.url and "i.redd.it" in submission.url and not is_image_url(submission.url):
        urls.append(submission.url + ".jpg")

    # Crosspost parent preview
    try:
        if getattr(submission, "crosspost_parent_list", None):
            parent = submission.crosspost_parent_list[0]
            if "preview" in parent:
                u = parent["preview"]["images"][0]["source"]["url"]
                u = unescape(u)
                if is_image_url(u):
                    urls.append(u)
    except Exception:
        pass

    return dedup(urls)


def dedup(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_filename(base: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def slug(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip())


def download_images(submission, out_dir: pathlib.Path) -> List[str]:
    urls = extract_image_urls(submission)
    saved: List[str] = []
    for idx, u in enumerate(urls, start=1):
        try:
            r = requests.get(u, timeout=20, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            # Pick extension from URL if present, else fall back to .jpg
            parsed = urlparse(u)
            ext = pathlib.Path(parsed.path).suffix.lower() or ".jpg"
            fn = safe_filename(f"{submission.id}_{idx}{ext}")
            out_path = out_dir / fn
            with open(out_path, "wb") as f:
                f.write(r.content)
            saved.append(str(out_path))
        except Exception as e:
            print(f"Failed image {u}: {e}")
    return saved


def harvest_subreddit(name: str, limit: Optional[int] = None, write_csv: bool = False, search_query: Optional[str] = None) -> None:
    """
    Collect newest submissions from a subreddit, or search results if search_query is set.
    Uses a simple checkpoint by last seen id to avoid reprocessing.
    """
    state = load_state()
    state_key = f"{name}|{search_query}" if search_query else name
    last_id = state["last_ids"].get(state_key)

    if search_query:
        out_jsonl = DATA_DIR / f"search_{slug(search_query)}.jsonl"
        out_csv = DATA_DIR / f"search_{slug(search_query)}.csv"
    else:
        out_jsonl = DATA_DIR / f"submissions_{name}.jsonl"
        out_csv = DATA_DIR / f"submissions_{name}.csv"

    count = 0
    new_last_id = last_id

    # Choose iterator: subreddit.new() or subreddit.search()
    if search_query:
        iterator = reddit.subreddit(name).search(search_query, sort="new", time_filter=TIME_FILTER, limit=limit)
    else:
        iterator = reddit.subreddit(name).new(limit=limit)

    # Iterate newest first; stop once we hit the previous last_id
    for s in iterator:
        if s.id == last_id:
            break

        # Skip NSFW images unless allowed
        if SAVE_MEDIA and (SAVE_NSFW or not getattr(s, "over_18", False)):
            # Decide subdirectory layout
            if MEDIA_SUBDIR_MODE == "subreddit":
                subdir = MEDIA_DIR / name
            elif MEDIA_SUBDIR_MODE == "query":
                subdir = MEDIA_DIR / (slug(search_query) if search_query else name)
            elif MEDIA_SUBDIR_MODE == "subreddit_then_query":
                subdir = MEDIA_DIR / name / (slug(search_query) if search_query else "_no_query")
            else:
                subdir = MEDIA_DIR / name
            subdir.mkdir(parents=True, exist_ok=True)
            saved_paths = download_images(s, subdir)
        else:
            saved_paths = []

        rec = submission_to_dict(s)
        # record the image source URLs for provenance, even if media saving is disabled
        try:
            source_urls = extract_image_urls(s)
        except Exception:
            source_urls = []
        if source_urls:
            rec["image_source_urls"] = source_urls
        if search_query:
            rec["search_query"] = search_query
        if saved_paths:
            rec["image_paths"] = saved_paths

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1
        if new_last_id is None:
            new_last_id = s.id
        time.sleep(0.3)  # be polite

    # Update checkpoint only if we saw anything
    if new_last_id and new_last_id != last_id:
        state["last_ids"][state_key] = new_last_id
        save_state(state)

    # Optional CSV write from JSONL
    if write_csv and out_jsonl.exists():
        try:
            import pandas as pd
            df = pd.read_json(out_jsonl, lines=True)
            df.to_csv(out_csv, index=False)
        except Exception as e:
            print("CSV write failed:", e)

    stage = f"search '{search_query}' in r/{name}" if search_query else f"r/{name}"
    print(f"Harvested {count} new items from {stage}")


def main() -> None:
    # Build list of queries if SEARCH_QUERIES is provided
    queries = None
    if SEARCH_QUERIES:
        queries = [q.strip() for q in re.split(r"[;,]", SEARCH_QUERIES) if q.strip()]

    has_queries = bool(queries or SEARCH_QUERY)

    # MODE A: Combine new() from SUBREDDITS + search() in r/all for queries
    if has_queries and COMBINE_NEW_AND_ALL_SEARCH:
        # Pass 1: newest posts from each named subreddit
        for name in SUBREDDITS:
            tries = 0
            while True:
                try:
                    harvest_subreddit(name, limit=LIMIT, write_csv=WRITE_CSV, search_query=None)
                    break
                except ResponseException as e:
                    if getattr(e, "response", None) and e.response.status_code == 429:
                        tries += 1
                        wait = min(60 * tries, 300)
                        print(f"Hit 429. Sleeping {wait} seconds...")
                        time.sleep(wait)
                        continue
                    raise
                except (RequestException, Forbidden) as e:
                    tries += 1
                    wait = min(30 * tries, 180)
                    print(f"Transient error {e}. Sleeping {wait} seconds...")
                    time.sleep(wait)
                    continue
        # Pass 2: search r/all for each query
        all_queries = queries if queries else [SEARCH_QUERY]
        for q in all_queries:
            tries = 0
            while True:
                try:
                    harvest_subreddit("all", limit=LIMIT, write_csv=WRITE_CSV, search_query=q)
                    break
                except ResponseException as e:
                    if getattr(e, "response", None) and e.response.status_code == 429:
                        tries += 1
                        wait = min(60 * tries, 300)
                        print(f"Hit 429. Sleeping {wait} seconds...")
                        time.sleep(wait)
                        continue
                    raise
                except (RequestException, Forbidden) as e:
                    tries += 1
                    wait = min(30 * tries, 180)
                    print(f"Transient error {e}. Sleeping {wait} seconds...")
                    time.sleep(wait)
                    continue
        return

    # MODE B: Queries only → search r/all only (no per-subreddit searches)
    if has_queries:
        all_queries = queries if queries else [SEARCH_QUERY]
        for q in all_queries:
            tries = 0
            while True:
                try:
                    harvest_subreddit("all", limit=LIMIT, write_csv=WRITE_CSV, search_query=q)
                    break
                except ResponseException as e:
                    if getattr(e, "response", None) and e.response.status_code == 429:
                        tries += 1
                        wait = min(60 * tries, 300)
                        print(f"Hit 429. Sleeping {wait} seconds...")
                        time.sleep(wait)
                        continue
                    raise
                except (RequestException, Forbidden) as e:
                    tries += 1
                    wait = min(30 * tries, 180)
                    print(f"Transient error {e}. Sleeping {wait} seconds...")
                    time.sleep(wait)
                    continue
        return

    # MODE C: No queries → just pull new() from the named subreddits
    for name in SUBREDDITS:
        tries = 0
        while True:
            try:
                harvest_subreddit(name, limit=LIMIT, write_csv=WRITE_CSV, search_query=None)
                break
            except ResponseException as e:
                if getattr(e, "response", None) and e.response.status_code == 429:
                    tries += 1
                    wait = min(60 * tries, 300)
                    print(f"Hit 429. Sleeping {wait} seconds...")
                    time.sleep(wait)
                    continue
                raise
            except (RequestException, Forbidden) as e:
                tries += 1
                wait = min(30 * tries, 180)
                print(f"Transient error {e}. Sleeping {wait} seconds...")
                time.sleep(wait)
                continue


if __name__ == "__main__":
    main()
