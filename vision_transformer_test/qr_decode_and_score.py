import argparse, os, warnings, csv, re, unicodedata, ipaddress
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

# ----- Optional decoders -----
try:
    import zxingcpp as zxing
    HAVE_ZXING = True
except Exception:
    HAVE_ZXING = False

try:
    from pyzbar.pyzbar import decode as zb_decode
    HAVE_PYZBAR = True
except Exception:
    HAVE_PYZBAR = False

# Silence QReader's encoding warnings; load lazily in second pass
warnings.filterwarnings("ignore", message="Double decoding failed", module="qreader")
try:
    from qreader import QReader
    HAVE_QREADER = True
except Exception:
    HAVE_QREADER = False

try:
    from charset_normalizer import from_bytes as cn_from_bytes
    HAVE_CN = True
except Exception:
    HAVE_CN = False

# ----- Helpers -----
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

def load_bgr_with_exif(path: Path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB","RGBA","L"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def load_rgb(path: Path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

# ----- OpenCV detect/decode & crops -----
_DET = None
def get_cv_detector():
    global _DET
    if _DET is None:
        _DET = cv2.QRCodeDetector()
    return _DET

def try_opencv_decode(img_bgr):
    det = get_cv_detector()
    out = []
    try:
        ok, texts, _, _ = det.detectAndDecodeMulti(img_bgr)
        if ok and texts is not None:
            out += [t for t in texts if t]
    except Exception:
        pass
    try:
        t, _, _ = det.detectAndDecode(img_bgr)
        if t:
            out.append(t)
    except Exception:
        pass
    return uniquify(out)

def try_opencv_detect_quads(img_bgr):
    det = get_cv_detector()
    quads = []
    try:
        ok, _, pts, _ = det.detectAndDecodeMulti(img_bgr)
        if pts is not None and len(pts) > 0:
            quads += [p.reshape(4,2) for p in pts]
    except Exception:
        pass
    try:
        ok, pts = det.detect(img_bgr)
        if ok and pts is not None and len(pts) > 0:
            quads += [p.reshape(4,2) for p in pts]
    except Exception:
        pass
    uniq, seen = [], set()
    for q in quads:
        key = tuple(map(int, np.round(q).ravel()))
        if key not in seen:
            uniq.append(q); seen.add(key)
    return uniq

def warp_crop_from_quad(img_bgr, quad, pad=0.12, out_size=640):
    q = np.array(quad, dtype=np.float32).reshape(-1,2)
    s = q.sum(axis=1); d = np.diff(q, axis=1).ravel()
    tl = q[np.argmin(s)]; br = q[np.argmax(s)]
    tr = q[np.argmin(d)]; bl = q[np.argmax(d)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    w = np.linalg.norm(tr - tl); h = np.linalg.norm(bl - tl)
    padw = pad*w; padh = pad*h
    src_pad = src.copy()
    src_pad[0] = [tl[0]-padw, tl[1]-padh]
    src_pad[1] = [tr[0]+padw, tr[1]-padh]
    src_pad[2] = [br[0]+padw, br[1]+padh]
    src_pad[3] = [bl[0]-padw, bl[1]+padh]
    dst = np.array([[0,0],[out_size,0],[out_size,out_size],[0,out_size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pad, dst)
    return cv2.warpPerspective(img_bgr, M, (out_size, out_size))

def clean_patch(patch_bgr):
    g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

# ----- ZXing & Pyzbar -----
def try_zxing(img_bgr):
    if not HAVE_ZXING:
        return []
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = zxing.read_barcodes(rgb)
        return uniquify([r.text for r in res if getattr(r, "text", None)])
    except Exception:
        return []

def best_effort_decode_bytes(b: bytes):
    try:
        return b.decode("utf-8")
    except Exception:
        if HAVE_CN:
            return str(cn_from_bytes(b).best())
        try:
            return b.decode("latin-1", errors="replace")
        except Exception:
            return b.decode("utf-8", errors="replace")

def try_pyzbar_on_pil(pil_img):
    if not HAVE_PYZBAR:
        return []
    out = []
    try:
        for d in zb_decode(pil_img):
            out.append(best_effort_decode_bytes(d.data))
    except Exception:
        pass
    return uniquify(out)

def try_pyzbar_from_path(path):
    if not HAVE_PYZBAR:
        return []
    try:
        return try_pyzbar_on_pil(Image.open(path))
    except Exception:
        return []

# ----- First pass: CPU (no QReader in workers) -----
def decode_one_cpu(path_str: str):
    p = Path(path_str)
    try:
        img = load_bgr_with_exif(p)
    except Exception:
        return {"path": str(p), "texts": [], "error": "unreadable_image"}

    # full-frame tries (prefer ZXing for ECI)
    texts = try_zxing(img)
    if not texts:
        texts = try_opencv_decode(img)
    if not texts:
        texts = try_pyzbar_from_path(str(p))

    # crops if still empty
    if not texts:
        quads = try_opencv_detect_quads(img)
        cropped = []
        for q in quads:
            patch = clean_patch(warp_crop_from_quad(img, q, pad=0.12, out_size=640))
            cropped += try_zxing(patch)
            if not cropped:
                cropped += try_opencv_decode(patch)
            if not cropped:
                rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                cropped += try_pyzbar_on_pil(Image.fromarray(rgb))
        texts = uniquify(cropped)

    texts = [sanitize(t, keep_newlines=True) for t in texts if t]
    return {"path": str(p), "texts": texts, "error": "" if texts else "no_qr_found_or_decoded"}

# ----- Second pass: QReader on GPU for misses -----
_QR = None
def get_qreader():
    if not HAVE_QREADER:
        return None
    global _QR
    if _QR is None:
        _QR = QReader()  # uses CUDA if torch sees it
        try:
            import torch
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    return _QR

def decode_with_qreader(path_str: str):
    qr = get_qreader()
    if qr is None:
        return []
    try:
        arr = load_rgb(Path(path_str))
        res = qr.detect_and_decode(image=arr) or []
        return uniquify([sanitize(t) for t in res if t])
    except Exception:
        return []

# ----- URL extraction & risk scoring -----
URL_REGEX = re.compile(r'(?i)\b((?:https?://|www\.)[^\s<>()"]+)')
SHORTENERS = {"bit.ly","t.co","tinyurl.com","goo.gl","ow.ly","is.gd","rebrand.ly","cutt.ly","rb.gy","shrtco.de","lnkd.in"}
ABUSED_TLDS = {"tk","ml","ga","cf","gq","xyz","top","club","work","click","live","cam","shop","ru","cn"}
BAD_PATH = {"login","verify","account","secure","update","wallet","reset","signin","auth","bank","gift","prize"}
BAD_QS = {"redirect","url","next","dest","destination","continue","r","u","q","link","target"}

def extract_urls(text):
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

# ----- Main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder with QR images")
    ap.add_argument("--out-decoded", required=True, help="Per-image decoded output")
    ap.add_argument("--out-labeled", required=True, help="Per-code per-URL risk scores")
    ap.add_argument("--out-wide", default="decoded_qr_wide.csv", help="Optional wide table")
    ap.add_argument("--wide", action="store_true", help="Also write the wide table")
    ap.add_argument("--workers", type=int, default=80, help="CPU workers for first pass")
    args = ap.parse_args()

    paths = sorted(walk_images(Path(args.folder)))
    if not paths:
        raise SystemExit(f"No images found under {args.folder}")

    # Quiet OpenCV logs
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try: cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception: pass

    # First pass: CPU pool (no QReader)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(tqdm(ex.map(decode_one_cpu, paths, chunksize=24), total=len(paths), desc="CPU pass"))

    # Second pass: QReader on GPU for misses
    misses = [r for r in rows if not r["texts"]]
    if misses and HAVE_QREADER:
        qr = get_qreader()
        for r in tqdm(misses, desc="QReader pass"):
            t = decode_with_qreader(r["path"])
            if t:
                r["texts"] = t
                r["error"] = ""

    # Build decoded dataframe
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

    # ----- risk scoring per code per URL -----
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
