"""
Downloads the DICOMs listed in rsna_mapping.csv from the mirror, converts
each to an 8-bit JPEG, and stores them as
data/rsna_chest_xray/{split}/{label}/{patientId}.jpg
so config.DATA_DIR = "./data/rsna_chest_xray" plugs straight into the
existing pipeline (train/ and test/ each contain NORMAL/ and PNEUMONIA/).
Resumable (skips existing .jpg), atomic writes, retry x4, failure log.
Run: python rsna_download.py --limit 6      # smoke test first!
Then: python rsna_download.py               # full run
Needs: pip install pydicom
"""
import os
import io
import argparse
import requests
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

HF_BASE = "https://huggingface.co/datasets/Baldezo313/rsna-pneumonia-dataset/resolve/main"
OUT_ROOT = os.path.join("data", "rsna_chest_xray")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def dcm_to_jpeg(data: bytes) -> bytes:
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float64)
    if arr.ndim == 3:                      # a few RSNA files are RGB
        arr = arr.mean(axis=-1)
    arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) \
        + float(getattr(ds, "RescaleIntercept", 0.0))
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def fetch_one(row):
    # pid, shard, label, split = row["patientId"], row["shard"], row["label"], row["split"]
    pid, shard, label, split = row.patientId, row.shard, row.label, row.split
    dst_dir = os.path.join(OUT_ROOT, split, label)
    dst = os.path.join(dst_dir, f"{pid}.jpg")
    if os.path.exists(dst):
        return "skip"
    url = f"{HF_BASE}/{shard}/{pid}.dcm"
    for _ in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            jpg = dcm_to_jpeg(r.content)
            os.makedirs(dst_dir, exist_ok=True)
            with open(dst + ".part", "wb") as f:
                f.write(jpg)
            os.replace(dst + ".part", dst)   # atomic: no half-written files
            return "ok"
        except Exception:
            continue
    return f"fail:{pid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke-test: only first N rows")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv("rsna_mapping.csv")
    if args.limit:
        df = df.head(args.limit)

    fails = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_one, row) for row in df.itertuples()]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            res = fut.result()
            if res.startswith("fail"):
                fails.append(res[5:])
    if fails:
        with open("rsna_download_failures.txt", "w") as f:
            f.write("\n".join(fails))
        print(f"{len(fails)} failures logged to rsna_download_failures.txt")

    for split in ("train", "test"):
        for label in ("NORMAL", "PNEUMONIA"):
            d = os.path.join(OUT_ROOT, split, label)
            n = len([f for f in os.listdir(d) if f.endswith(".jpg")]) if os.path.isdir(d) else 0
            print(f"{split}/{label}: {n} jpg on disk")


if __name__ == "__main__":
    main()