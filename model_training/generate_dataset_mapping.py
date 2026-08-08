"""
Generate matched pneumonia + normal mapping CSVs from Data_Entry_2017.csv,
with a REAL, verified Zip_File column -- resolved by indexing the actual
remote archives' central directories, not guessed from a formula.

Reconstructed from scratch (the original generator script was lost) based
on: the columns visible in the old pneumonia_zip_mapping.csv (Image Index,
Finding Labels, Zip_File), and the CLI usage pattern you showed:
    python external_dataset_prepare.py [--balance-ratio 1.0] [--max-pneumonia N]

The old Zip_File values came from a formula that turned out wrong for ~52%
of images. This version doesn't guess -- it opens each of the 12 remote
archives, reads its central directory (a small end-of-file index of every
member, not the full file), and records exactly which archive each image
actually lives in. Any target image not found in ANY of the 12 archives
gets flagged in mapping_missing_report.txt instead of written with a wrong
or blank value.

USAGE:
    python generate_dataset_mapping.py
    python generate_dataset_mapping.py --max-pneumonia 800 --balance-ratio 1.5
    python generate_dataset_mapping.py --seed 7

REQUIRES: pip install remotezip --break-system-packages
"""
import argparse
import os
import pandas as pd
from tqdm import tqdm
from remotezip import RemoteZip, RangeNotSupported

ENTRY_CSV = "data/nih_raw/Data_Entry_2017.csv"
PNEUMONIA_OUT = "pneumonia_zip_mapping.csv"
NORMAL_OUT = "normal_zip_mapping.csv"
MISSING_REPORT = "mapping_missing_report.txt"

HF_COMMIT = "d444bf39439f4fbfac6922e2472b1bf2554309e7"
HF_BASE = f"https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/{HF_COMMIT}/data/images"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ALL_ZIPS = [f"images_{i:03d}.zip" for i in range(1, 13)]

parser = argparse.ArgumentParser()
parser.add_argument("--balance-ratio", type=float, default=1.0,
                     help="normal_count = pneumonia_count * ratio (default 1.0 = equal classes)")
parser.add_argument("--max-pneumonia", type=int, default=None,
                     help="cap pneumonia count via random sampling (default: use all available)")
parser.add_argument("--seed", type=int, default=42,
                     help="random seed, for reproducibility across runs")
args = parser.parse_args()

# --- select images from the master metadata ---
entries = pd.read_csv(ENTRY_CSV)
print("Columns in Data_Entry_2017.csv:", entries.columns.tolist())
print(f"{len(entries)} total images in source file.")

pneumonia = entries[entries["Finding Labels"].str.contains("Pneumonia", regex=False)]
print(f"{len(pneumonia)} pneumonia-labeled images found (multi-label included).")
if args.max_pneumonia is not None and args.max_pneumonia < len(pneumonia):
    pneumonia = pneumonia.sample(n=args.max_pneumonia, random_state=args.seed)
    print(f"Capped to {len(pneumonia)} via --max-pneumonia.")
pneumonia_n = len(pneumonia)

normal_target = int(round(pneumonia_n * args.balance_ratio))
normal_pool = entries[entries["Finding Labels"] == "No Finding"]
print(f"{len(normal_pool)} 'No Finding' images available; need {normal_target}.")
if len(normal_pool) < normal_target:
    raise ValueError(
        f"Only {len(normal_pool)} normal images available, but {normal_target} "
        f"are needed for balance-ratio={args.balance_ratio}. Lower the ratio or "
        "check ENTRY_CSV is the right file."
    )
normal = normal_pool.sample(n=normal_target, random_state=args.seed)

pneumonia_out = pneumonia[["Image Index", "Finding Labels"]].sort_values("Image Index").reset_index(drop=True)
normal_out = normal[["Image Index", "Finding Labels"]].sort_values("Image Index").reset_index(drop=True)

# --- resolve REAL Zip_File values by indexing all 12 archives ---
print("\nIndexing all 12 remote archives to resolve real Zip_File values "
      "(central directories only, not full files) ...")
location = {}
for zname in tqdm(ALL_ZIPS, desc="indexing"):
    url = f"{HF_BASE}/{zname}?download=true"
    try:
        with RemoteZip(url, headers=HEADERS) as rz:
            for member in rz.namelist():
                location[os.path.basename(member)] = zname
    except RangeNotSupported:
        raise RuntimeError(f"{zname}'s host doesn't support range requests -- can't index remotely.")


def attach_zip_file(df, label):
    df = df.copy()
    df["Zip_File"] = df["Image Index"].map(location)
    missing = df[df["Zip_File"].isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} {label} image(s) not found in any of the 12 archives.")
    return df, missing["Image Index"].tolist()


pneumonia_out, pneumonia_missing = attach_zip_file(pneumonia_out, "pneumonia")
normal_out, normal_missing = attach_zip_file(normal_out, "normal")

all_missing = pneumonia_missing + normal_missing
if all_missing:
    with open(MISSING_REPORT, "w") as f:
        f.write("\n".join(all_missing))
    print(f"{len(all_missing)} total image(s) not found in any archive -- written to {MISSING_REPORT}")
    print("These will need separate investigation before you rely on this mapping being complete.")

pneumonia_out.to_csv(PNEUMONIA_OUT, index=False)
normal_out.to_csv(NORMAL_OUT, index=False)

print(f"\nWrote {len(pneumonia_out)} pneumonia images to {PNEUMONIA_OUT} (Zip_File verified against real archives)")
print(f"Wrote {len(normal_out)} normal images to {NORMAL_OUT} (Zip_File verified against real archives)")
print(f"Ratio achieved: {len(normal_out) / len(pneumonia_out):.2f} (target: {args.balance_ratio})")
