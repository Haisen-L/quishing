#!/usr/bin/env python3
"""
qr_offline_build_vector_db_8models_matched.py

Offline vector database builder for QR only images.
The model names and default checkpoints match the user's retrieval testing script:
DINOv2, ConvNeXtV2_FCMAE, CvT, Swin, DINOv3, PE, SigLIP2, EVA02.

Purpose
-------
1. Load labeled QR image metadata from CSV.
2. Select the gallery split.
3. Generate L2 normalized image embeddings for each selected model.
4. Build one FAISS cosine similarity index per model.
5. Save model specific indexes plus shared gallery metadata.

CSV columns
-----------
Required:
    sample_id
    label
    split

Image path column:
    The script auto detects qr_crop_path first, then qr_real_path.
    You can also specify --image_col manually.

Typical use
-----------
python qr_offline_build_vector_db_8models_matched.py \
  --csv qr_crop_fit_L_prepared_712.csv \
  --output_dir offline_vector_db_qr_only_8models \
  --gallery_split gallery \
  --image_col qr_crop_path \
  --device cuda \
  --batch_size_dinov2 2048 \
  --batch_size_convnext 1024 \
  --batch_size_cvt 1024 \
  --batch_size_swin 1024 \
  --batch_size_dinov3 2048 \
  --batch_size_pe 2048 \
  --batch_size_siglip2 2048 \
  --batch_size_eva02 2048 \
  --num_workers 8 \
  --prefetch_factor 4
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import faiss
except Exception:
    faiss = None

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


def resolve_image_column(df: pd.DataFrame, image_col: str) -> str:
    if image_col and image_col.lower() != "auto":
        if image_col not in df.columns:
            raise ValueError(f"Image column '{image_col}' was not found. Available columns: {list(df.columns)}")
        return image_col

    for candidate in ["qr_crop_path", "qr_real_path", "image_path", "path"]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not auto detect the image path column. "
        "Expected one of qr_crop_path, qr_real_path, image_path, or path. "
        "Use --image_col to specify it manually."
    )


def make_absolute_paths(paths: Sequence[str], image_root: Optional[str]) -> List[str]:
    root = Path(image_root).expanduser().resolve() if image_root else None
    resolved = []
    for p in paths:
        p_obj = Path(str(p)).expanduser()
        if p_obj.is_absolute() or root is None:
            resolved.append(str(p_obj))
        else:
            resolved.append(str(root / p_obj))
    return resolved


def sample_if_requested(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    if "label" in df.columns and df["label"].nunique() > 1:
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
        self.ckpt = ckpt
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
        store_dtype: torch.dtype,
        desc: str,
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
        self.ckpt = ckpt
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
        store_dtype: torch.dtype,
        desc: str,
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
    def __init__(self, ckpt: str, hf_fallback_ckpt: str, device: str):
        self.ckpt = ckpt
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
                errors.append(f"open_clip create_model_and_transforms({ckpt}): {e}")
                self.backend = None

            try:
                self.backend = "open_clip"
                self.model, self.preprocess = open_clip.create_model_from_pretrained(ckpt, device=self.device)
                self.model.eval()
                return
            except Exception as e:
                errors.append(f"open_clip create_model_from_pretrained({ckpt}): {e}")
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
            "Unable to load PE Core B 16. Try installing open_clip_torch or use a newer timm or transformers version. "
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
        store_dtype: torch.dtype,
        desc: str,
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
    parser.add_argument("--image_col", type=str, default="auto")
    parser.add_argument("--image_root", type=str, default="")
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

    parser.add_argument("--max_gallery", type=int, default=0)
    parser.add_argument("--store_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--save_torch_embeddings", action="store_true")
    parser.add_argument("--save_numpy_embeddings", action="store_true")
    parser.add_argument("--check_files", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="DINOv2,ConvNeXtV2_FCMAE,CvT,Swin,DINOv3,PE,SigLIP2,EVA02",
        help="Comma separated subset of exact model names to run.",
    )
    return parser.parse_args()


def get_run_specs(args: argparse.Namespace) -> Dict[str, RunConfig]:
    return {
        "DINOv2": RunConfig("DINOv2", "DINOv2 on QR crops", args.batch_size_dinov2),
        "ConvNeXtV2_FCMAE": RunConfig("ConvNeXtV2_FCMAE", "ConvNeXt V2 with FCMAE on QR crops", args.batch_size_convnext),
        "CvT": RunConfig("CvT", "CvT on QR crops", args.batch_size_cvt),
        "Swin": RunConfig("Swin", "Swin on QR crops", args.batch_size_swin),
        "DINOv3": RunConfig("DINOv3", "DINOv3 on QR crops", args.batch_size_dinov3),
        "PE": RunConfig("PE", "PE Core B 16 on QR crops", args.batch_size_pe),
        "SigLIP2": RunConfig("SigLIP2", "SigLIP2 on QR crops", args.batch_size_siglip2),
        "EVA02": RunConfig("EVA02", "EVA 02 on QR crops", args.batch_size_eva02),
    }


def build_embedder(model_name: str, args: argparse.Namespace):
    if model_name == "DINOv2":
        return HFVisionEmbedder(args.dinov2_ckpt, args.device), args.dinov2_ckpt
    if model_name == "ConvNeXtV2_FCMAE":
        return TimmEmbedder(args.convnext_ckpt, args.device), args.convnext_ckpt
    if model_name == "CvT":
        return HFVisionEmbedder(args.cvt_ckpt, args.device), args.cvt_ckpt
    if model_name == "Swin":
        return TimmEmbedder(args.swin_ckpt, args.device), args.swin_ckpt
    if model_name == "DINOv3":
        return TimmEmbedder(args.dinov3_ckpt, args.device), args.dinov3_ckpt
    if model_name == "PE":
        return PEEmbedder(args.pe_ckpt, args.pe_hf_fallback_ckpt, args.device), args.pe_ckpt
    if model_name == "SigLIP2":
        return TimmEmbedder(args.siglip2_ckpt, args.device), args.siglip2_ckpt
    if model_name == "EVA02":
        return TimmEmbedder(args.eva02_ckpt, args.device), args.eva02_ckpt
    raise ValueError(f"Unknown model name: {model_name}")


def build_faiss_index(emb: torch.Tensor, output_path: Path) -> Dict[str, int]:
    if faiss is None:
        raise ImportError("faiss is required to build vector indexes. Install faiss cpu or faiss gpu.")
    emb_np = emb.float().contiguous().numpy().astype("float32", copy=False)
    dim = int(emb_np.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(emb_np)
    faiss.write_index(index, str(output_path))
    return {"dim": dim, "ntotal": int(index.ntotal)}


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    index_dir = outdir / "faiss_indexes"
    emb_dir = outdir / "embeddings"
    meta_dir = outdir / "metadata"
    index_dir.mkdir(exist_ok=True)
    emb_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    require_columns(df, ["sample_id", "label", "split"])
    image_col = resolve_image_column(df, args.image_col)

    df["sample_id"] = df["sample_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["split"] = df["split"].astype(str)

    gallery_df = df[df["split"] == args.gallery_split].reset_index(drop=True)
    if len(gallery_df) == 0:
        raise ValueError(f"No rows found for gallery split '{args.gallery_split}'.")

    gallery_df = sample_if_requested(gallery_df, args.max_gallery, args.seed)
    gallery_df["resolved_image_path"] = make_absolute_paths(gallery_df[image_col].tolist(), args.image_root or None)

    if args.check_files:
        missing = [p for p in gallery_df["resolved_image_path"].tolist() if not Path(p).exists()]
        if missing:
            preview = missing[:10]
            raise FileNotFoundError(f"Found {len(missing)} missing image files. First missing paths: {preview}")

    gallery_paths = gallery_df["resolved_image_path"].tolist()
    gallery_df.to_csv(meta_dir / "gallery_metadata.csv", index=False)

    all_run_specs = get_run_specs(args)
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in all_run_specs]
    if unknown:
        raise ValueError(f"Unknown model names in --models: {unknown}. Valid names: {list(all_run_specs.keys())}")

    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    manifest_rows = []

    for model_name in requested:
        run = all_run_specs[model_name]
        print(f"\n=== Building vector index for {model_name} ===")
        embedder, ckpt_used = build_embedder(model_name, args)

        gallery_emb = embedder.embed_paths(
            gallery_paths,
            batch_size=run.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            store_dtype=store_dtype,
            desc=f"Embedding gallery: {run.pretty_name}",
        )

        if gallery_emb.ndim != 2 or gallery_emb.shape[0] != len(gallery_df):
            raise RuntimeError(
                f"Unexpected embedding shape for {model_name}: {tuple(gallery_emb.shape)}. "
                f"Expected ({len(gallery_df)}, dim)."
            )

        if args.save_torch_embeddings:
            torch.save(gallery_emb, emb_dir / f"gallery_emb_{model_name}.pt")
        if args.save_numpy_embeddings:
            np.save(emb_dir / f"gallery_emb_{model_name}.npy", gallery_emb.float().numpy().astype("float32"))

        index_info = build_faiss_index(gallery_emb, index_dir / f"gallery_{model_name}.faiss")

        model_manifest = {
            "model_name": model_name,
            "checkpoint": ckpt_used,
            "pretty_name": run.pretty_name,
            "batch_size": run.batch_size,
            "gallery_split": args.gallery_split,
            "n_gallery": int(len(gallery_df)),
            "embedding_dim": index_info["dim"],
            "faiss_index_type": "IndexFlatIP",
            "similarity": "cosine similarity because embeddings are L2 normalized before indexing",
            "image_col": image_col,
            "metadata_file": "metadata/gallery_metadata.csv",
            "index_file": f"faiss_indexes/gallery_{model_name}.faiss",
        }
        with open(outdir / f"manifest_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(model_manifest, f, indent=2)
        manifest_rows.append(model_manifest)

        del embedder, gallery_emb
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    pd.DataFrame(manifest_rows).to_csv(outdir / "vector_db_manifest.csv", index=False)

    print("\nOffline vector database build complete.")
    print(f"Shared metadata: {meta_dir / 'gallery_metadata.csv'}")
    print(f"FAISS indexes: {index_dir}")
    print(f"Manifest: {outdir / 'vector_db_manifest.csv'}")


if __name__ == "__main__":
    main()
