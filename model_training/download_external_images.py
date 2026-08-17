"""
Download images for one or more class mapping CSVs (e.g. normal_zip_mapping.csv,
pneumonia_zip_mapping.csv), one dataset at a time -- each fully completes
before the next starts, so downloads for different classes never interleave.

Trusts the Zip_File column directly rather than re-indexing all 12 archives
on every run, since generate_dataset_mapping.py now resolves that column
against the real archives at generation time (not a formula guess). Only
opens the specific archive(s) each job actually needs.

To change the order (e.g. pneumonia before normal), just reorder JOBS below.

REQUIRES: pip install remotezip --break-system-packages
"""
import os
import pandas as pd
from tqdm import tqdm
from remotezip import RemoteZip, RangeNotSupported

HF_COMMIT = "d444bf39439f4fbfac6922e2472b1bf2554309e7"
HF_BASE = f"https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/{HF_COMMIT}/data/images"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# (label, csv_path, out_dir) -- runs top to bottom, one fully finishing
# before the next starts. Add more rows here for other finding classes.
JOBS = [
    ("normal",    "normal_zip_mapping.csv",    "data/external_nih_chestxray14/NORMAL"),
    ("pneumonia", "pneumonia_zip_mapping.csv", "data/external_nih_chestxray14/PNEUMONIA"),
    ("mix_normal",    "trainmix_normal_zip_mapping.csv", "data/nih_train_mix/NORMAL"),
    ("mix_pneumonia", "trainmix_pneumonia_zip_mapping.csv", "data/nih_train_mix/PNEUMONIA"),
]


def run_job(label, csv_path, out_dir):
    print(f"\n{'=' * 60}\n{label.upper()} -- {csv_path} -> {out_dir}\n{'=' * 60}")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "Zip_File" not in df.columns:
        raise ValueError(
            f"{csv_path} has no Zip_File column -- regenerate it with "
            "generate_dataset_mapping.py first."
        )

    needed = df[["Image Index", "Zip_File"]].drop_duplicates()
    still_needed = needed[~needed["Image Index"].apply(
        lambda f: os.path.exists(os.path.join(out_dir, f))
    )]
    print(f"{len(needed)} unique images needed, {len(still_needed)} not yet on disk.")

    if still_needed.empty:
        print(f"{label}: nothing to do, all files already present.")
        return

    total_bytes = 0
    failed = []
    for zname, group in still_needed.groupby("Zip_File"):
        url = f"{HF_BASE}/{zname}?download=true"
        print(f"Opening {zname} for {len(group)} file(s) ...")
        try:
            with RemoteZip(url, headers=HEADERS) as rz:
                members = {os.path.basename(m): m for m in rz.namelist()}
                for fname in tqdm(group["Image Index"], desc=f"{label}: {zname}"):
                    member = members.get(fname)
                    if member is None:
                        print(f"  NOT FOUND in {zname} (stale Zip_File? "
                              f"try regenerating the mapping CSV): {fname}")
                        failed.append(fname)
                        continue
                    dest = os.path.join(out_dir, fname)
                    try:
                        data = rz.read(member)
                    except Exception as e:
                        print(f"  FAILED to fetch {fname}: {e}")
                        failed.append(fname)
                        continue
                    with open(dest, "wb") as out:
                        out.write(data)
                    total_bytes += len(data)
        except RangeNotSupported:
            raise RuntimeError(f"{zname}'s host doesn't support range requests.")

    now_have = sum(1 for f in needed["Image Index"] if os.path.exists(os.path.join(out_dir, f)))
    print(f"{label}: ~{total_bytes / 1e6:.1f} MB transferred this run.")
    print(f"{label}: now have {now_have}/{len(needed)} images on disk.")
    if failed:
        report = f"{label}_download_failures.txt"
        with open(report, "w") as f:
            f.write("\n".join(failed))
        print(f"{label}: {len(failed)} file(s) failed -- see {report}")


for label, csv_path, out_dir in JOBS:
    run_job(label, csv_path, out_dir)

print("\nAll jobs complete.")