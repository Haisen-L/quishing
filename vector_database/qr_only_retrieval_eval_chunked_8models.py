#!/usr/bin/env python3
"""
qr_only_retrieval_eval_chunked_8models.py

Chunked QR-only retrieval evaluation for eight embedding backbones:
1) DINOv2
2) ConvNeXt V2 with FCMAE
3) CvT
4) Swin
5) DINOv3
6) PE-Core-B-16
7) SigLIP2
8) EVA-02

This script is designed for large QR-only galleries and queries.
It avoids materializing the full query x gallery similarity matrix by using
chunked retrieval over gallery and query embeddings.

Required CSV columns
--------------------
- sample_id
- label
- qr_crop_path

Optional CSV columns
--------------------
- split                # gallery / val / test. If missing, use --auto_split
- family_id            # optional relevance key for retrieval
- visible_percentage
- distortion_type
- logo_insertion
- campaign_family
- scene_type
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import timm
    from timm.data import resolve_model_data_config, create_transform
except Exception:
    timm = None
    resolve_model_data_config = None
    create_transform = None

try:
    from transformers import AutoImageProcessor, AutoModel
except Exception:
    AutoImageProcessor = None
    AutoModel = None

try:
    import open_clip
except Exception:
    open_clip = None


@dataclass
class RunConfig:
    model_name: str
    pretty_name: str
    batch_size: int


class PathDataset(Dataset):
    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> str:
        return self.paths[idx]


def pil_load(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def pil_collate(paths: List[str]) -> List[Image.Image]:
    return [pil_load(p) for p in paths]


def tensor_collate_factory(transform):
    def _collate(paths: List[str]) -> torch.Tensor:
        imgs = [transform(pil_load(p)) for p in paths]
        return torch.stack(imgs, dim=0)
    return _collate


def autocast_context(device: torch.device, enabled: bool = True):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts(dropna=False)
    return len(counts) > 1 and int(counts.min()) >= 2


def create_712_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    strat_labels = df["label"] if can_stratify(df["label"]) else None
    idx = np.arange(len(df))

    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=0.20,
        random_state=seed,
        stratify=strat_labels if strat_labels is not None else None,
    )
    df.loc[idx_test, "split"] = "test"

    trainval = df.iloc[idx_trainval].reset_index(drop=True)
    strat_tv = trainval["label"] if can_stratify(trainval["label"]) else None
    tv_idx = np.arange(len(trainval))

    idx_gallery_local, idx_val_local = train_test_split(
        tv_idx,
        test_size=0.125,
        random_state=seed,
        stratify=strat_tv if strat_tv is not None else None,
    )

    orig_gallery = idx_trainval[idx_gallery_local]
    orig_val = idx_trainval[idx_val_local]
    df.loc[orig_gallery, "split"] = "gallery"
    df.loc[orig_val, "split"] = "val"

    if df["split"].isna().any() or (df["split"] == "").any():
        raise RuntimeError("Failed to assign split to all rows.")
    return df


def stratified_sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df.reset_index(drop=True)

    if "label" in df.columns and can_stratify(df["label"]):
        sampled = (
            df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(n=max(1, int(round(n * len(g) / len(df)))), random_state=seed))
        )
        if len(sampled) > n:
            sampled = sampled.sample(n=n, random_state=seed)
        elif len(sampled) < n:
            remaining = df.drop(sampled.index)
            extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, extra], axis=0)
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df.sample(n=n, random_state=seed).reset_index(drop=True)


class HFVisionEmbedder:
    def __init__(self, ckpt: str, device: str):
        if AutoImageProcessor is None or AutoModel is None:
            raise ImportError("transformers is required for Hugging Face models.")
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()

    def build_loader(self, paths: Sequence[str], batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
        kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": pil_collate,
            "pin_memory": False,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(PathDataset(paths), **kwargs)

    def _pool_output(self, out) -> torch.Tensor:
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            x = out.last_hidden_state
            if x.ndim == 3:
                return x[:, 0]
            if x.ndim == 4:
                return x.mean(dim=(2, 3))
        if isinstance(out, (tuple, list)) and len(out) > 0:
            x = out[0]
            if x.ndim == 2:
                return x
            if x.ndim == 3:
                return x[:, 0]
            if x.ndim == 4:
                return x.mean(dim=(2, 3))
        raise RuntimeError("Unable to derive embedding from Hugging Face model output.")

    @torch.no_grad()
    def embed_paths(
        self,
        paths: Sequence[str],
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        store_dtype: torch.dtype = torch.float16,
        desc: str = "Embedding HF model",
    ) -> torch.Tensor:
        loader = self.build_loader(paths, batch_size, num_workers, prefetch_factor)
        outputs = []
        for imgs in tqdm(loader, desc=desc):
            batch = self.processor(images=imgs, return_tensors="pt")
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with autocast_context(self.device, enabled=True):
                out = self.model(**batch)
                emb = self._pool_output(out)
            emb = F.normalize(emb.float(), dim=1).to(store_dtype).cpu()
            outputs.append(emb)
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 1), dtype=store_dtype)


class TimmEmbedder:
    def __init__(self, ckpt: str, device: str):
        if timm is None:
            raise ImportError("timm is required for timm models.")
        self.device = torch.device(device)
        self.model = timm.create_model(ckpt, pretrained=True, num_classes=0).to(self.device).eval()
        cfg = resolve_model_data_config(self.model)
        self.transform = create_transform(**cfg, is_training=False)

    def build_loader(self, paths: Sequence[str], batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
        kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": tensor_collate_factory(self.transform),
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(PathDataset(paths), **kwargs)

    @torch.no_grad()
    def embed_paths(
        self,
        paths: Sequence[str],
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        store_dtype: torch.dtype = torch.float16,
        desc: str = "Embedding timm model",
    ) -> torch.Tensor:
        loader = self.build_loader(paths, batch_size, num_workers, prefetch_factor)
        outputs = []
        for x in tqdm(loader, desc=desc):
            x = x.to(self.device, non_blocking=True)
            with autocast_context(self.device, enabled=True):
                emb = self.model(x)
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                if emb.ndim == 4:
                    emb = emb.mean(dim=(2, 3))
            emb = F.normalize(emb.float(), dim=1).to(store_dtype).cpu()
            outputs.append(emb)
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 1), dtype=store_dtype)


class PEEmbedder:
    """Try multiple backends for PE-Core-B-16.

    Order:
    1) open_clip with hf-hub:timm/PE-Core-B-16
    2) timm with hf-hub:timm/PE-Core-B-16
    3) transformers with facebook/PE-Core-B16-224
    """
    def __init__(self, ckpt: str, hf_fallback_ckpt: str, device: str):
        self.device = torch.device(device)
        self.backend = None
        self.inner = None
        errors = []

        if open_clip is not None:
            try:
                self.backend = "open_clip"
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(ckpt, pretrained=None, device=self.device)
                self.model.eval()
                return
            except Exception as e:
                errors.append(f"open_clip({ckpt}): {e}")
                self.backend = None

            try:
                self.backend = "open_clip"
                self.model, self.preprocess = open_clip.create_model_from_pretrained(ckpt, device=self.device)
                self.model.eval()
                return
            except Exception as e:
                errors.append(f"open_clip pretrained({ckpt}): {e}")
                self.backend = None

        if timm is not None:
            try:
                self.backend = "timm"
                self.inner = TimmEmbedder(ckpt, device)
                return
            except Exception as e:
                errors.append(f"timm({ckpt}): {e}")
                self.backend = None

        if AutoImageProcessor is not None and AutoModel is not None:
            try:
                self.backend = "hf"
                self.inner = HFVisionEmbedder(hf_fallback_ckpt, device)
                return
            except Exception as e:
                errors.append(f"transformers({hf_fallback_ckpt}): {e}")
                self.backend = None

        raise RuntimeError(
            "Unable to load PE-Core-B-16. Try installing open_clip_torch or use a newer timm/transformers version. Errors: "
            + " | ".join(errors)
        )

    def build_loader(self, paths: Sequence[str], batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
        if self.backend in {"timm", "hf"}:
            return self.inner.build_loader(paths, batch_size, num_workers, prefetch_factor)
        kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": tensor_collate_factory(self.preprocess),
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(PathDataset(paths), **kwargs)

    @torch.no_grad()
    def embed_paths(
        self,
        paths: Sequence[str],
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        store_dtype: torch.dtype = torch.float16,
        desc: str = "Embedding PE",
    ) -> torch.Tensor:
        if self.backend in {"timm", "hf"}:
            return self.inner.embed_paths(paths, batch_size, num_workers, prefetch_factor, store_dtype, desc)
        loader = self.build_loader(paths, batch_size, num_workers, prefetch_factor)
        outputs = []
        for x in tqdm(loader, desc=desc):
            x = x.to(self.device, non_blocking=True)
            with autocast_context(self.device, enabled=True):
                emb = self.model.encode_image(x)
            emb = F.normalize(emb.float(), dim=1).to(store_dtype).cpu()
            outputs.append(emb)
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 1), dtype=store_dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gallery_split", type=str, default="gallery")
    parser.add_argument("--query_split", type=str, default="test")
    parser.add_argument("--auto_split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dinov2_ckpt", type=str, default="facebook/dinov2-base")
    parser.add_argument("--convnext_ckpt", type=str, default="convnextv2_base.fcmae_ft_in22k_in1k")
    parser.add_argument("--cvt_ckpt", type=str, default="microsoft/cvt-21")
    parser.add_argument("--swin_ckpt", type=str, default="swin_base_patch4_window7_224.ms_in22k_ft_in1k")
    parser.add_argument("--dinov3_ckpt", type=str, default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--pe_ckpt", type=str, default="hf-hub:timm/PE-Core-B-16")
    parser.add_argument("--pe_hf_fallback_ckpt", type=str, default="facebook/PE-Core-B16-224")
    parser.add_argument("--siglip2_ckpt", type=str, default="vit_base_patch16_siglip_256.v2_webli")
    parser.add_argument("--eva02_ckpt", type=str, default="eva02_base_patch14_224.mim_in22k")

    parser.add_argument("--batch_size_dinov2", type=int, default=512)
    parser.add_argument("--batch_size_convnext", type=int, default=512)
    parser.add_argument("--batch_size_cvt", type=int, default=512)
    parser.add_argument("--batch_size_swin", type=int, default=512)
    parser.add_argument("--batch_size_dinov3", type=int, default=512)
    parser.add_argument("--batch_size_pe", type=int, default=512)
    parser.add_argument("--batch_size_siglip2", type=int, default=512)
    parser.add_argument("--batch_size_eva02", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    parser.add_argument("--gallery_chunk_size", type=int, default=65536)
    parser.add_argument("--query_chunk_size", type=int, default=512)

    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--risk_threshold", type=float, default=0.5)
    parser.add_argument("--relevance_mode", type=str, default="label", choices=["label", "family"])
    parser.add_argument("--malicious_label", type=str, default="malicious")
    parser.add_argument("--benign_label", type=str, default="benign")

    parser.add_argument("--max_gallery", type=int, default=0)
    parser.add_argument("--max_query", type=int, default=0)
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="DINOv2,ConvNeXtV2_FCMAE,CvT,Swin,DINOv3,PE,SigLIP2,EVA02",
        help="Comma-separated subset of models to run.",
    )
    return parser.parse_args()


def get_total_relevant_counts(gallery_df: pd.DataFrame, relevance_mode: str) -> Dict[str, int]:
    if relevance_mode == "family":
        if "family_id" not in gallery_df.columns:
            raise ValueError("family_id is required when relevance_mode='family'.")
        return gallery_df["family_id"].astype(str).value_counts().to_dict()
    return gallery_df["label"].astype(str).value_counts().to_dict()


def ap_at_k(relevance_topk: np.ndarray, total_relevant: int, k: int) -> float:
    denom = max(1, min(total_relevant, k))
    hits = 0
    ap = 0.0
    for i, rel in enumerate(relevance_topk[:k], start=1):
        if rel:
            hits += 1
            ap += hits / i
    return ap / denom


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def merge_topk(existing_sims, existing_idx, new_sims, new_idx, k):
    all_sims = torch.cat([existing_sims, new_sims], dim=1)
    all_idx = torch.cat([existing_idx, new_idx], dim=1)
    top_sims, pos = torch.topk(all_sims, k=min(k, all_sims.shape[1]), dim=1)
    top_idx = torch.gather(all_idx, 1, pos)
    if top_sims.shape[1] < k:
        pad_cols = k - top_sims.shape[1]
        top_sims = torch.cat([top_sims, torch.full((top_sims.shape[0], pad_cols), -float("inf"), device=top_sims.device)], dim=1)
        top_idx = torch.cat([top_idx, torch.full((top_idx.shape[0], pad_cols), -1, dtype=top_idx.dtype, device=top_idx.device)], dim=1)
    return top_sims, top_idx


@torch.no_grad()
def chunked_retrieval(
    gallery_emb: torch.Tensor,
    query_emb: torch.Tensor,
    gallery_df: pd.DataFrame,
    query_df: pd.DataFrame,
    device: torch.device,
    topk_values: Sequence[int],
    gallery_chunk_size: int,
    query_chunk_size: int,
    relevance_mode: str,
    malicious_label: str,
    benign_label: str,
    risk_threshold: float,
    model_name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    max_k = max(topk_values)
    benign_topn = 5

    gallery_labels = gallery_df["label"].astype(str).tolist()
    gallery_key = gallery_df["family_id"].astype(str).tolist() if relevance_mode == "family" else gallery_labels
    total_rel_counts = get_total_relevant_counts(gallery_df, relevance_mode)

    gallery_labels_arr = np.array(gallery_labels, dtype=object)
    gallery_key_arr = np.array(gallery_key, dtype=object)

    malicious_mask_full = torch.tensor([lbl == malicious_label for lbl in gallery_labels], dtype=torch.bool)
    benign_mask_full = torch.tensor([lbl == benign_label for lbl in gallery_labels], dtype=torch.bool)

    recall_scores = {k: [] for k in topk_values}
    precision_scores = {k: [] for k in topk_values}
    ap_scores = {k: [] for k in topk_values}
    top1_acc = []
    risk_true, risk_scores = [], []

    per_query_rows = []

    for q_start in tqdm(range(0, len(query_df), query_chunk_size), desc=f"Retrieval: {model_name}"):
        q_end = min(q_start + query_chunk_size, len(query_df))
        q_chunk_cpu = query_emb[q_start:q_end]
        q_chunk = q_chunk_cpu.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32, non_blocking=True)
        bq = q_chunk.shape[0]

        global_top_sims = torch.full((bq, max_k), -float("inf"), device=device)
        global_top_idx = torch.full((bq, max_k), -1, dtype=torch.long, device=device)

        global_top_benign = torch.full((bq, benign_topn), -float("inf"), device=device)
        global_max_mal = torch.full((bq,), -float("inf"), device=device)

        for g_start in range(0, len(gallery_df), gallery_chunk_size):
            g_end = min(g_start + gallery_chunk_size, len(gallery_df))
            g_chunk_cpu = gallery_emb[g_start:g_end]
            g_chunk = g_chunk_cpu.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32, non_blocking=True)

            sims = q_chunk @ g_chunk.T

            local_k = min(max_k, sims.shape[1])
            local_top_sims, local_pos = torch.topk(sims, k=local_k, dim=1)
            local_top_idx = local_pos + g_start
            global_top_sims, global_top_idx = merge_topk(global_top_sims, global_top_idx, local_top_sims, local_top_idx, max_k)

            mal_mask_chunk = malicious_mask_full[g_start:g_end].to(device)
            if mal_mask_chunk.any():
                mal_sims = sims[:, mal_mask_chunk]
                local_max_mal = mal_sims.max(dim=1).values
                global_max_mal = torch.maximum(global_max_mal, local_max_mal)

            ben_mask_chunk = benign_mask_full[g_start:g_end].to(device)
            if ben_mask_chunk.any():
                ben_sims = sims[:, ben_mask_chunk]
                local_bk = min(benign_topn, ben_sims.shape[1])
                local_top_ben, _ = torch.topk(ben_sims, k=local_bk, dim=1)
                if local_top_ben.shape[1] < benign_topn:
                    pad = benign_topn - local_top_ben.shape[1]
                    local_top_ben = torch.cat([local_top_ben, torch.full((bq, pad), -float("inf"), device=device)], dim=1)
                dummy_idx = torch.full_like(local_top_ben, -1, dtype=torch.long, device=device)
                global_top_benign, _ = merge_topk(global_top_benign, torch.full_like(global_top_benign, -1, dtype=torch.long), local_top_ben, dummy_idx, benign_topn)

            del g_chunk, sims

        top_sims_np = global_top_sims.cpu().float().numpy()
        top_idx_np = global_top_idx.cpu().numpy()
        top_ben_np = global_top_benign.cpu().float().numpy()
        max_mal_np = global_max_mal.cpu().float().numpy()

        for local_i, (_, qrow) in enumerate(query_df.iloc[q_start:q_end].iterrows()):
            idxs = top_idx_np[local_i]
            sims = top_sims_np[local_i]

            valid = idxs >= 0
            idxs = idxs[valid]
            sims = sims[valid]

            retrieved_labels = gallery_labels_arr[idxs].tolist()
            retrieved_keys = gallery_key_arr[idxs].tolist()

            query_key = str(qrow["family_id"]) if relevance_mode == "family" else str(qrow["label"])
            total_relevant = int(total_rel_counts.get(query_key, 0))
            rel_top = np.array([rk == query_key for rk in retrieved_keys], dtype=np.int32)

            for k in topk_values:
                rel_k = rel_top[:k]
                recall_scores[k].append(float(rel_k.sum() / max(1, total_relevant)))
                precision_scores[k].append(float(rel_k.mean()) if len(rel_k) > 0 else 0.0)
                ap_scores[k].append(ap_at_k(rel_top, total_relevant, k))

            top1_acc.append(int(retrieved_labels[0] == str(qrow["label"])) if len(retrieved_labels) else 0)

            top5_labels = retrieved_labels[:5]
            malicious_ratio = float(np.mean(np.array(top5_labels, dtype=object) == malicious_label)) if top5_labels else 0.0

            max_mal = float(max_mal_np[local_i]) if np.isfinite(max_mal_np[local_i]) else 0.0
            top_ben = top_ben_np[local_i]
            top_ben = top_ben[np.isfinite(top_ben) & (top_ben > -1e20)]
            anomaly = float(1.0 - top_ben.mean()) if len(top_ben) > 0 else 1.0

            risk = float(np.clip((0.4 * malicious_ratio) + (0.4 * max_mal) + (0.2 * anomaly), 0.0, 1.0))
            y_true = int(str(qrow["label"]) == malicious_label)
            y_pred = int(risk >= risk_threshold)

            risk_true.append(y_true)
            risk_scores.append(risk)

            per_query_rows.append({
                "sample_id": qrow["sample_id"],
                "model": model_name,
                "view": "qr_crop",
                "label": qrow["label"],
                "top1_label": retrieved_labels[0] if retrieved_labels else None,
                "top1_similarity": float(sims[0]) if len(sims) else None,
                "top5_labels": json.dumps(retrieved_labels[:5]),
                "top10_labels": json.dumps(retrieved_labels[:10]),
                "top5_similarities": json.dumps([round(float(x), 6) for x in sims[:5].tolist()]),
                "top10_similarities": json.dumps([round(float(x), 6) for x in sims[:10].tolist()]),
                "malicious_ratio": malicious_ratio,
                "max_malicious_similarity": max_mal,
                "anomaly_score": anomaly,
                "risk_score": risk,
                "risk_pred": y_pred,
            })

        del q_chunk, global_top_sims, global_top_idx, global_top_benign, global_max_mal

    y_true_arr = np.array(risk_true)
    y_score_arr = np.array(risk_scores)
    y_pred_arr = (y_score_arr >= risk_threshold).astype(int)

    summary = {
        "model": model_name,
        "view": "qr_crop",
        "n_queries": int(len(query_df)),
        "Top1_Accuracy": float(np.mean(top1_acc)) if top1_acc else float("nan"),
        "ROC_AUC": safe_roc_auc(y_true_arr, y_score_arr),
        "PR_AUC": safe_pr_auc(y_true_arr, y_score_arr),
        "F1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "Risk_Threshold": float(risk_threshold),
    }

    for k in topk_values:
        summary[f"Recall@{k}"] = float(np.mean(recall_scores[k])) if recall_scores[k] else float("nan")
        summary[f"Precision@{k}"] = float(np.mean(precision_scores[k])) if precision_scores[k] else float("nan")
        summary[f"mAP@{k}"] = float(np.mean(ap_scores[k])) if ap_scores[k] else float("nan")

    try:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        summary["TN"] = int(cm[0, 0])
        summary["FP"] = int(cm[0, 1])
        summary["FN"] = int(cm[1, 0])
        summary["TP"] = int(cm[1, 1])
    except Exception:
        pass

    return pd.DataFrame(per_query_rows), summary


def stratified_summary(per_query_df: pd.DataFrame, query_df: pd.DataFrame, output_dir: Path, model_name: str) -> None:
    merged = per_query_df.merge(query_df, on=["sample_id", "label"], how="left")
    rows = []

    if "visible_percentage" in merged.columns:
        merged["visibility_bin"] = np.where(
            pd.to_numeric(merged["visible_percentage"], errors="coerce").fillna(-1) >= 50,
            "high_visibility",
            "low_visibility",
        )
        for group_name, g in merged.groupby("visibility_bin"):
            rows.append({
                "group_type": "visibility",
                "group_name": group_name,
                "n": int(len(g)),
                "avg_risk": float(g["risk_score"].mean()),
                "avg_top1_similarity": float(g["top1_similarity"].mean()),
            })

    if "distortion_type" in merged.columns:
        for group_name, g in merged.groupby(merged["distortion_type"].fillna("unknown")):
            rows.append({
                "group_type": "distortion_type",
                "group_name": str(group_name),
                "n": int(len(g)),
                "avg_risk": float(g["risk_score"].mean()),
                "avg_top1_similarity": float(g["top1_similarity"].mean()),
            })

    if "logo_insertion" in merged.columns:
        for group_name, g in merged.groupby(merged["logo_insertion"].fillna("unknown")):
            rows.append({
                "group_type": "logo_insertion",
                "group_name": str(group_name),
                "n": int(len(g)),
                "avg_risk": float(g["risk_score"].mean()),
                "avg_top1_similarity": float(g["top1_similarity"].mean()),
            })

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / f"stratified_{model_name}_qr_crop.csv", index=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    require_columns(df, ["sample_id", "label", "qr_crop_path"])
    df["sample_id"] = df["sample_id"].astype(str)
    df["label"] = df["label"].astype(str)

    if "split" not in df.columns:
        if not args.auto_split:
            raise ValueError("CSV has no split column. Use --auto_split to create a 7:1:2 split.")
        df = create_712_split(df, seed=args.seed)
        df.to_csv(outdir / "dataset_with_auto_split.csv", index=False)
    else:
        df["split"] = df["split"].astype(str)

    gallery_df = df[df["split"] == args.gallery_split].reset_index(drop=True)
    query_df = df[df["split"] == args.query_split].reset_index(drop=True)

    if len(gallery_df) == 0:
        raise ValueError(f"No rows found for gallery split '{args.gallery_split}'.")
    if len(query_df) == 0:
        raise ValueError(f"No rows found for query split '{args.query_split}'.")

    gallery_df = stratified_sample(gallery_df, args.max_gallery if args.max_gallery > 0 else None, args.seed)
    query_df = stratified_sample(query_df, args.max_query if args.max_query > 0 else None, args.seed)

    gallery_paths = gallery_df["qr_crop_path"].tolist()
    query_paths = query_df["qr_crop_path"].tolist()

    all_run_specs = {
        "DINOv2": RunConfig("DINOv2", "DINOv2 on QR crops", args.batch_size_dinov2),
        "ConvNeXtV2_FCMAE": RunConfig("ConvNeXtV2_FCMAE", "ConvNeXt V2 with FCMAE on QR crops", args.batch_size_convnext),
        "CvT": RunConfig("CvT", "CvT on QR crops", args.batch_size_cvt),
        "Swin": RunConfig("Swin", "Swin on QR crops", args.batch_size_swin),
        "DINOv3": RunConfig("DINOv3", "DINOv3 on QR crops", args.batch_size_dinov3),
        "PE": RunConfig("PE", "PE-Core-B-16 on QR crops", args.batch_size_pe),
        "SigLIP2": RunConfig("SigLIP2", "SigLIP2 on QR crops", args.batch_size_siglip2),
        "EVA02": RunConfig("EVA02", "EVA-02 on QR crops", args.batch_size_eva02),
    }

    requested = [m.strip() for m in args.models.split(',') if m.strip()]
    unknown = [m for m in requested if m not in all_run_specs]
    if unknown:
        raise ValueError(f"Unknown model names in --models: {unknown}")
    runs = [all_run_specs[m] for m in requested]

    device = torch.device(args.device)
    embedders = {
        "DINOv2": HFVisionEmbedder(args.dinov2_ckpt, args.device),
        "ConvNeXtV2_FCMAE": TimmEmbedder(args.convnext_ckpt, args.device),
        "CvT": HFVisionEmbedder(args.cvt_ckpt, args.device),
        "Swin": TimmEmbedder(args.swin_ckpt, args.device),
        "DINOv3": TimmEmbedder(args.dinov3_ckpt, args.device),
        "PE": PEEmbedder(args.pe_ckpt, args.pe_hf_fallback_ckpt, args.device),
        "SigLIP2": TimmEmbedder(args.siglip2_ckpt, args.device),
        "EVA02": TimmEmbedder(args.eva02_ckpt, args.device),
    }

    all_summaries, all_per_query = [], []

    for run in runs:
        embedder = embedders[run.model_name]

        gallery_emb = embedder.embed_paths(
            gallery_paths,
            batch_size=run.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            store_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            desc=f"Embedding gallery: {run.pretty_name}",
        )
        query_emb = embedder.embed_paths(
            query_paths,
            batch_size=run.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            store_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            desc=f"Embedding query: {run.pretty_name}",
        )

        if args.save_embeddings:
            torch.save(gallery_emb, outdir / f"gallery_emb_{run.model_name}_qr_crop.pt")
            torch.save(query_emb, outdir / f"query_emb_{run.model_name}_qr_crop.pt")

        per_query_df, summary = chunked_retrieval(
            gallery_emb=gallery_emb,
            query_emb=query_emb,
            gallery_df=gallery_df,
            query_df=query_df,
            device=device,
            topk_values=args.topk,
            gallery_chunk_size=args.gallery_chunk_size,
            query_chunk_size=args.query_chunk_size,
            relevance_mode=args.relevance_mode,
            malicious_label=args.malicious_label,
            benign_label=args.benign_label,
            risk_threshold=args.risk_threshold,
            model_name=run.model_name,
        )

        per_query_df.to_csv(outdir / f"per_query_{run.model_name}_qr_crop.csv", index=False)
        stratified_summary(per_query_df, query_df, outdir, run.model_name)

        all_per_query.append(per_query_df)
        all_summaries.append(summary)

        del gallery_emb, query_emb
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(outdir / "summary_metrics_qr_only_chunked_8models.csv", index=False)

    if all_per_query:
        pd.concat(all_per_query, ignore_index=True).to_csv(
            outdir / "all_per_query_outputs_qr_only_chunked_8models.csv", index=False
        )

    if args.max_gallery > 0 or args.max_query > 0:
        gallery_df.to_csv(outdir / "gallery_subset_used.csv", index=False)
        query_df.to_csv(outdir / "query_subset_used.csv", index=False)

    print("\nSaved:")
    print(f"- {outdir / 'summary_metrics_qr_only_chunked_8models.csv'}")
    print(f"- {outdir / 'all_per_query_outputs_qr_only_chunked_8models.csv'}")
    if (outdir / "gallery_subset_used.csv").exists():
        print(f"- {outdir / 'gallery_subset_used.csv'}")
        print(f"- {outdir / 'query_subset_used.csv'}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
