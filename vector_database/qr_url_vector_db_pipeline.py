#!/usr/bin/env python3
"""
qr_url_vector_db_pipeline.py

Build and test a URL embedding vector database for the QR phishing project.

This script adds the URL/Text branch to your existing QR image vector database system.
It does not open, crawl, or visit any URL. It only embeds the URL string from the CSV.

Input CSV can be either:
1. The master image CSV, for example qr_crop_fit_L_prepared_712.csv
2. The URL only CSV, for example qr_url_prepared_712.csv

Required columns:
    sample_id, label, split, url

Typical full run:
python qr_url_vector_db_pipeline.py \
  --csv qr_url_prepared_712.csv \
  --output_dir offline_vector_db_qr_only_8models \
  --query_output_dir url_vector_db_test_results \
  --gallery_split gallery \
  --query_split test \
  --text_model sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 4096 \
  --topk 10 \
  --risk_threshold 0.5 \
  --device cuda

Install:
    pip install sentence-transformers faiss-cpu pandas numpy scikit-learn tqdm
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Sequence, Tuple
from urllib.parse import urlparse, parse_qsl, unquote

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    raise SystemExit(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from exc


TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV containing sample_id, label, split, and url.")
    parser.add_argument("--output_dir", required=True, help="Existing or new vector database root folder.")
    parser.add_argument("--query_output_dir", default="url_vector_db_test_results", help="Folder for URL query results.")
    parser.add_argument("--gallery_split", default="gallery")
    parser.add_argument("--query_split", default="test")
    parser.add_argument("--url_col", default="url")
    parser.add_argument("--sample_id_col", default="sample_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--faiss_query_batch", type=int, default=8192)
    parser.add_argument("--risk_threshold", type=float, default=0.5)
    parser.add_argument("--malicious_label", default="malicious")
    parser.add_argument("--benign_label", default="benign")
    parser.add_argument("--max_gallery", type=int, default=0, help="Optional debug limit. 0 means all gallery rows.")
    parser.add_argument("--max_query", type=int, default=0, help="Optional debug limit. 0 means all query rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--rebuild_index", action="store_true")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_tokens(text: str) -> List[str]:
    tokens = [t for t in TOKEN_SPLIT_RE.split(text.lower()) if t]
    return tokens


def normalize_url_for_embedding(url_value) -> str:
    """Convert a URL into a text string useful for a general text embedding model.

    The model sees domain tokens, path tokens, query keys, and simple lexical features.
    This improves URL similarity without visiting the URL.
    """
    raw = safe_text(url_value)
    raw_unquoted = unquote(raw)
    raw_for_parse = raw_unquoted

    if raw_for_parse and "://" not in raw_for_parse:
        raw_for_parse = "http://" + raw_for_parse

    parsed = urlparse(raw_for_parse)
    scheme = parsed.scheme.lower() if parsed.scheme else "none"
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    domain_tokens = split_tokens(host)
    path_tokens = split_tokens(path)
    query_pairs = parse_qsl(query, keep_blank_values=True)
    query_keys = [safe_text(k).lower() for k, _ in query_pairs if safe_text(k)]
    query_value_tokens = []
    for _, v in query_pairs[:10]:
        query_value_tokens.extend(split_tokens(str(v))[:5])

    raw_tokens = split_tokens(raw_unquoted)

    has_ip = "yes" if IP_RE.match(host) else "no"
    has_at = "yes" if "@" in raw_unquoted else "no"
    has_https = "yes" if scheme == "https" else "no"
    has_query = "yes" if query else "no"
    has_fragment = "yes" if parsed.fragment else "no"
    num_digits = sum(ch.isdigit() for ch in raw_unquoted)
    num_dots = raw_unquoted.count(".")
    num_hyphens = raw_unquoted.count("-")
    num_slashes = raw_unquoted.count("/")
    length = len(raw_unquoted)

    text_parts = [
        f"scheme {scheme}",
        f"https {has_https}",
        f"host {' '.join(domain_tokens) if domain_tokens else 'none'}",
        f"path {' '.join(path_tokens) if path_tokens else 'none'}",
        f"query_keys {' '.join(query_keys) if query_keys else 'none'}",
        f"query_values {' '.join(query_value_tokens) if query_value_tokens else 'none'}",
        f"has_ip {has_ip}",
        f"has_at_symbol {has_at}",
        f"has_query {has_query}",
        f"has_fragment {has_fragment}",
        f"length {length}",
        f"digits {num_digits}",
        f"dots {num_dots}",
        f"hyphens {num_hyphens}",
        f"slashes {num_slashes}",
        f"raw_tokens {' '.join(raw_tokens[:50]) if raw_tokens else 'none'}",
    ]
    return " | ".join(text_parts)


def encode_texts_to_mmap(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int,
    output_path: Path,
    desc: str,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    if n == 0:
        raise ValueError("No texts to encode.")

    first_batch = list(texts[: min(batch_size, n)])
    first_emb = model.encode(
        first_batch,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    dim = first_emb.shape[1]
    mmap = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=(n, dim))
    mmap[: first_emb.shape[0]] = first_emb

    start = first_emb.shape[0]
    for start in tqdm(range(start, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        emb = model.encode(
            list(texts[start:end]),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        mmap[start:end] = emb

    mmap.flush()
    return np.load(output_path, mmap_mode="r")


def build_faiss_index(embeddings: np.ndarray, index_path: Path, chunk_size: int = 100000) -> faiss.Index:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    for start in tqdm(range(0, len(embeddings), chunk_size), desc="Building URL FAISS index"):
        end = min(start + chunk_size, len(embeddings))
        block = np.ascontiguousarray(embeddings[start:end].astype("float32"))
        faiss.normalize_L2(block)
        index.add(block)
    faiss.write_index(index, str(index_path))
    return index


def faiss_search(index: faiss.Index, query_emb: np.ndarray, topk: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    all_sims = []
    all_idxs = []
    for start in tqdm(range(0, len(query_emb), batch_size), desc="Searching URL index"):
        end = min(start + batch_size, len(query_emb))
        q = np.ascontiguousarray(query_emb[start:end].astype("float32"))
        faiss.normalize_L2(q)
        sims, idxs = index.search(q, topk)
        all_sims.append(sims)
        all_idxs.append(idxs)
    return np.vstack(all_sims), np.vstack(all_idxs)


def metric_summary(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    unique = np.unique(y_true)
    return {
        "n_queries": int(len(y_true)),
        "risk_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(unique) > 1 else math.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(unique) > 1 else math.nan,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    vector_db_dir = Path(args.output_dir)
    query_out_dir = Path(args.query_output_dir)
    url_index_dir = vector_db_dir / "url_faiss_indexes"
    url_meta_dir = vector_db_dir / "url_metadata"
    url_emb_dir = vector_db_dir / "url_embeddings"
    query_out_dir.mkdir(parents=True, exist_ok=True)
    url_index_dir.mkdir(parents=True, exist_ok=True)
    url_meta_dir.mkdir(parents=True, exist_ok=True)
    url_emb_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    require_columns(df, [args.sample_id_col, args.label_col, args.split_col, args.url_col])
    df = df.copy()
    df[args.sample_id_col] = df[args.sample_id_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)
    df[args.split_col] = df[args.split_col].astype(str)
    df[args.url_col] = df[args.url_col].apply(safe_text)
    df["url_embedding_text"] = df[args.url_col].apply(normalize_url_for_embedding)

    gallery_df = df[df[args.split_col] == args.gallery_split].copy().reset_index(drop=True)
    query_df = df[df[args.split_col] == args.query_split].copy().reset_index(drop=True)

    if args.max_gallery and args.max_gallery > 0 and args.max_gallery < len(gallery_df):
        gallery_df = gallery_df.sample(n=args.max_gallery, random_state=args.seed).reset_index(drop=True)
    if args.max_query and args.max_query > 0 and args.max_query < len(query_df):
        query_df = query_df.sample(n=args.max_query, random_state=args.seed).reset_index(drop=True)

    if len(gallery_df) == 0:
        raise ValueError(f"No gallery rows found for split {args.gallery_split}")
    if len(query_df) == 0:
        raise ValueError(f"No query rows found for split {args.query_split}")

    print(f"Gallery rows: {len(gallery_df):,}")
    print(f"Query rows:   {len(query_df):,}")
    print(f"Text model:   {args.text_model}")

    model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.text_model.split("/")[-1])
    index_path = url_index_dir / f"gallery_URL_{model_tag}.faiss"
    gallery_emb_path = url_emb_dir / f"gallery_URL_{model_tag}.npy"
    query_emb_path = url_emb_dir / f"query_{args.query_split}_URL_{model_tag}.npy"
    gallery_meta_path = url_meta_dir / "gallery_url_metadata.csv"

    model = SentenceTransformer(args.text_model, device=args.device)

    if index_path.exists() and gallery_meta_path.exists() and not args.rebuild_index:
        print(f"Loading existing URL index: {index_path}")
        index = faiss.read_index(str(index_path))
        gallery_meta = pd.read_csv(gallery_meta_path)
    else:
        print("Encoding gallery URLs")
        gallery_emb = encode_texts_to_mmap(
            model=model,
            texts=gallery_df["url_embedding_text"].tolist(),
            batch_size=args.batch_size,
            output_path=gallery_emb_path,
            desc="Encoding gallery URLs",
        )
        gallery_meta = gallery_df[[args.sample_id_col, args.label_col, args.split_col, args.url_col]].copy()
        gallery_meta = gallery_meta.rename(
            columns={
                args.sample_id_col: "sample_id",
                args.label_col: "label",
                args.split_col: "split",
                args.url_col: "url",
            }
        )
        gallery_meta["url_embedding_text"] = gallery_df["url_embedding_text"]
        gallery_meta["url_text_model"] = args.text_model
        gallery_meta.to_csv(gallery_meta_path, index=False)
        index = build_faiss_index(gallery_emb, index_path)

    if index.ntotal != len(gallery_meta):
        raise ValueError(f"URL index and metadata mismatch: index={index.ntotal}, metadata={len(gallery_meta)}")

    print("Encoding query URLs")
    query_emb = encode_texts_to_mmap(
        model=model,
        texts=query_df["url_embedding_text"].tolist(),
        batch_size=args.batch_size,
        output_path=query_emb_path,
        desc="Encoding query URLs",
    )

    sims, idxs = faiss_search(index, query_emb, args.topk, args.faiss_query_batch)

    gallery_labels = gallery_meta["label"].astype(str).to_numpy()
    gallery_ids = gallery_meta["sample_id"].astype(str).to_numpy()
    gallery_urls = gallery_meta["url"].astype(str).to_numpy()

    rows = []
    risk_scores = []
    y_true = (query_df[args.label_col].astype(str).to_numpy() == args.malicious_label).astype(int)

    for i in range(len(query_df)):
        neighbor_idx = idxs[i]
        neighbor_labels = gallery_labels[neighbor_idx].astype(str).tolist()
        neighbor_ids = gallery_ids[neighbor_idx].astype(str).tolist()
        neighbor_urls = gallery_urls[neighbor_idx].astype(str).tolist()
        malicious_ratio = float(np.mean(np.array(neighbor_labels) == args.malicious_label))
        risk_scores.append(malicious_ratio)
        rows.append(
            {
                "sample_id": str(query_df.loc[i, args.sample_id_col]),
                "model": "URL_TextEncoder",
                "text_model": args.text_model,
                "label": str(query_df.loc[i, args.label_col]),
                "url": str(query_df.loc[i, args.url_col]),
                "top1_sample_id": neighbor_ids[0],
                "top1_label": neighbor_labels[0],
                "top1_url": neighbor_urls[0],
                "top1_similarity": float(sims[i, 0]),
                f"top{args.topk}_sample_ids": json.dumps(neighbor_ids),
                f"top{args.topk}_labels": json.dumps(neighbor_labels),
                f"top{args.topk}_similarities": json.dumps([round(float(x), 6) for x in sims[i].tolist()]),
                "malicious_ratio": malicious_ratio,
                "risk_score": malicious_ratio,
                "risk_pred": int(malicious_ratio >= args.risk_threshold),
            }
        )

    risk_scores = np.array(risk_scores, dtype=float)
    per_query_df = pd.DataFrame(rows)
    per_query_path = query_out_dir / f"per_query_URL_TextEncoder_{args.query_split}.csv"
    per_query_df.to_csv(per_query_path, index=False)

    summary = metric_summary(y_true, risk_scores, args.risk_threshold)
    summary.update(
        {
            "model": "URL_TextEncoder",
            "text_model": args.text_model,
            "topk": int(args.topk),
            "gallery_rows": int(len(gallery_meta)),
            "query_split": args.query_split,
            "index_file": str(index_path),
            "metadata_file": str(gallery_meta_path),
        }
    )
    summary_df = pd.DataFrame([summary])
    summary_path = query_out_dir / f"summary_URL_TextEncoder_{args.query_split}.csv"
    summary_json_path = query_out_dir / f"summary_URL_TextEncoder_{args.query_split}.json"
    summary_df.to_csv(summary_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "branch": "URL/Text embedding branch",
        "text_model": args.text_model,
        "similarity": "cosine similarity using L2 normalized embeddings and FAISS IndexFlatIP",
        "does_not_fetch_urls": True,
        "gallery_split": args.gallery_split,
        "query_split": args.query_split,
        "url_index_file": str(index_path),
        "url_metadata_file": str(gallery_meta_path),
        "per_query_file": str(per_query_path),
        "summary_file": str(summary_path),
    }
    (vector_db_dir / "manifest_URL_TextEncoder.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.save_embeddings:
        # Keep index and metadata as the main vector database artifacts.
        # The embeddings can be large and are optional.
        # Remove query embeddings, keep gallery embeddings only if user requested saving.
        try:
            query_emb_path.unlink(missing_ok=True)
            gallery_emb_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("\nSaved URL vector database artifacts:")
    print(f"- {index_path}")
    print(f"- {gallery_meta_path}")
    print(f"- {vector_db_dir / 'manifest_URL_TextEncoder.json'}")
    print("\nSaved URL query results:")
    print(f"- {per_query_path}")
    print(f"- {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
