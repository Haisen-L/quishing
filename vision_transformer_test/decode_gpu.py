import argparse, os, warnings, csv, re, unicodedata, ipaddress
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# QReader only
warnings.filterwarnings("ignore", message="Double decoding failed", module="qreader")
try:
    from qreader import QReader
    HAVE_QREADER = True
except Exception:
    HAVE_QREADER = False

# optional charset normalization for byte decoding safety
try:
    from charset_normalizer import from_bytes as cn_from_bytes
    HAVE_CN = True
except Exception:
    HAVE_CN = False


# ---------- helpers ----------
def sanitize(t: str, keep_newlines=True) -> str:
    if not isinstance(t, str):
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\\r\\n", "\n")
    if keep_newlines:
        t = re.sub(r"[^\S\n]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
    else:
        t = re.sub(r"\s+", " ", t)
    t = unicodedata.normalize("NFC", t)
    return t.strip()

def uniquify(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out

def walk_images(root: Path):
    exts = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield str(p)

def load_rgb(path: Path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

# ---------- GPU decoder ----------
_QR = None
def get_qreader(require_cuda: bool = True):
    if not HAVE_QREADER:
        raise RuntimeError("QReader is not installed. Please `pip install qreader`.")
    global _QR
    if _QR is None:
        # ensure CUDA is present if required
        try:
            import torch
            if require_cuda and not torch.cuda.is_available():
                raise RuntimeError("CUDA not available but GPU was required. Install CUDA torch or use a GPU machine.")
            torch.backends.cudnn.benchmark = True
        except Exception as e:
            if require_cuda:
                raise
        _QR = QReader()  # QReader uses CUDA when torch sees it
    return _QR

def decode_with_qreader(path_str: str) -> list[str]:
    try:
        arr = load_rgb(Path(path_str))
    except Exception:
        return []
    try:
        res = get_qreader().detect_and_decode(image=arr) or []
        return uniquify([sanitize(t) for t in res if t])
    except Exception:
        return []

# ---------- URL extraction and risk scoring ----------
URL_REGEX = re.compile(r'(?i)\b((?:https?://|www\.)[^\s<>()"]+)')
SHORTENERS = {"bit.ly","t.co","tinyurl.com","goo.gl","ow.ly","is.gd","rebrand.ly","cutt.ly","rb.gy","shrtco.de","lnkd.in"}
ABUSED_TLDS = {"tk","ml","ga","cf","gq","xyz","top","club","work","click","live","cam","shop","ru","cn"}
BAD_PATH = {"login","verify","account","secure","update","wallet","reset","signin","auth","bank","gift","prize"}
BAD_QS = {"redirect","url","next","dest","destination","continue","r","u","q","link","target"}

def extract_urls(text: str):
    out = []
    for m in URL_REGEX.findall(text or ""):
        u = m.strip().rstrip(').,;]')
        if u.lower().startswith("www."):
            u = "http://" + u
        out.append(u)
    return out

def is_ip(host: str):
    try:
        ipaddress.ip_address(host); return True
    except Exception:
        return False

def score_url(u: str):
    reasons = []
    try:
        p = urlparse(u)
        scheme = (p.scheme or "").lower()
        host = (p.hostname or "").lower()
        score = 0

        if scheme in {"javascript","data"}:
            return 80, [f"dangerous_scheme:{scheme}"]
        if scheme == "http":
            score += 5; reasons.append("plaintext_http")
        if not host:
            score += 15; reasons.append("missing_host")
        else:
            parts = host.split(".")
            tld = parts[-1] if len(parts) >= 2 else ""
            if is_ip(host):
                score += 15; reasons.append("ip_host")
            if "xn--" in host:
                score += 20; reasons.append("punycode_idn")
            if tld in ABUSED_TLDS:
                score += 10; reasons.append(f"abused_tld:.{tld}")
            if len(parts) >= 4:
                score += 10; reasons.append("many_subdomains")
            if len(host) > 50:
                score += 5; reasons.append("long_host")
            if host in SHORTENERS:
                score += 25; reasons.append("shortener_domain")
        if "@" in (p.netloc or ""):
            score += 20; reasons.append("userinfo_in_netloc")
        if p.port and p.port not in {80,443}:
            score += 10; reasons.append(f"uncommon_port:{p.port}")
        path = (p.path or "").lower()
        if any(w in path for w in BAD_PATH):
            score += 10; reasons.append("suspicious_path_word")
        if path.count("//") >= 2:
            score += 5; reasons.append("path_double_slashes")
        qs = parse_qs(p.query, keep_blank_values=True)
        if any(k.lower() in BAD_QS for k in qs):
            score += 10; reasons.append("suspicious_query_key")
        if any(any("http://" in v or "https://" in v for v in vs) for vs in qs.values()):
            score += 15; reasons.append("nested_url_in_query")
        if len(u) > 120:
            score += 5; reasons.append("long_url")
        return score, reasons
    except Exception as e:
        return 0, [f"parse_error:{type(e).__name__}"]

def decide(score: int):
    return "likely_malicious" if score >= 50 else "suspicious" if score >= 30 else "benign"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with QR images")
    ap.add_argument("--out-decoded", required=True, help="Per image decoded output")
    ap.add_argument("--out-labeled", required=True, help="Per code per URL risk scores")
    ap.add_argument("--out-wide", default="decoded_qr_wide.csv", help="Optional wide table")
    ap.add_argument("--wide", action="store_true", help="Also write the wide table")
    ap.add_argument("--require-cuda", action="store_true", default=True, help="Fail if CUDA is not available")
    args = ap.parse_args()

    paths = sorted(walk_images(Path(args.folder)))
    if not paths:
        raise SystemExit(f"No images found under {args.folder}")

    # initialize QReader and assert CUDA if requested
    get_qreader(require_cuda=args.require_cuda)

    rows = []
    for p in paths:
        try:
            texts = decode_with_qreader(p)
            rows.append({
                "path": p,
                "texts": texts,
                "error": "" if texts else "no_qr_found_or_decoded"
            })
        except Exception as e:
            rows.append({
                "path": p,
                "texts": [],
                "error": f"decode_error:{type(e).__name__}"
            })

    decoded_df = pd.DataFrame({
        "path": [r["path"] for r in rows],
        "ok": [bool(r["texts"]) for r in rows],
        "num_texts": [len(r["texts"]) for r in rows],
        "texts_joined": [" | ".join(r["texts"]) for r in rows],
        "error": [r["error"] for r in rows],
    })
    decoded_df.to_csv(
        args.out_decoded, index=False, encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    if args.wide:
        max_n = max((len(r["texts"]) for r in rows), default=0)
        wide = decoded_df[["path","ok","num_texts","error"]].copy()
        for i in range(max_n):
            wide[f"text_{i+1}"] = [r["texts"][i] if i < len(r["texts"]) else "" for r in rows]
        wide.to_csv(
            args.out_wide, index=False, encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL, lineterminator="\n"
        )

    # risk scoring
    labeled_rows = []
    for r in rows:
        texts = r.get("texts", []) or []
        if not texts:
            labeled_rows.append({
                "path": r["path"], "qr_index": None, "raw_text": "",
                "url_index": None, "url": "", "risk_score": 0,
                "decision": "no_qr", "reasons": ""
            })
            continue
        for qi, piece in enumerate(texts, start=1):
            urls = extract_urls(piece)
            if not urls:
                labeled_rows.append({
                    "path": r["path"], "qr_index": qi, "raw_text": piece,
                    "url_index": 0, "url": "", "risk_score": 0,
                    "decision": "no_url", "reasons": ""
                })
            else:
                for ui, u in enumerate(urls, start=1):
                    sc, rs = score_url(u)
                    labeled_rows.append({
                        "path": r["path"], "qr_index": qi, "raw_text": piece,
                        "url_index": ui, "url": u, "risk_score": sc,
                        "decision": decide(sc), "reasons": ";".join(rs)
                    })

    labeled_df = pd.DataFrame(labeled_rows)
    labeled_df.to_csv(
        args.out_labeled, index=False, encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL, lineterminator="\n"
    )

    print(f"Saved {args.out_decoded} and {args.out_labeled}" + (f" and {args.out_wide}" if args.wide else ""))

if __name__ == "__main__":
    main()

