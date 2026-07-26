"""
Pull the pneumonia images out of the 12 remote zip archives using HTTP
range requests, indexing every archive's REAL contents first instead of
trusting the CSV's Zip_File column.

WHY: the last run showed 739/1431 images (52%) missing from the archive
the CSV said they'd be in. The CSV's Zip_File assignment looks like it was
computed from an assumed image-index formula that doesn't match the real
archive boundaries on this mirror -- images_012.zip had zero misses, which
fits files spilling into the *next* zip over from where the CSV expected
them. So: stop trusting that column, and instead ask each archive directly
what it actually contains.

Phase 1 fetches only each archive's central directory (a small end-of-file
index of every member + its byte offset) to build a real filename -> zip
lookup across all 12 archives.
Phase 2 fetches only the compressed bytes for the images you actually need,
from wherever they actually are.

REQUIRES: pip install remotezip --break-system-packages
"""
import os
import pandas as pd
from tqdm import tqdm
from remotezip import RemoteZip, RangeNotSupported

CSV_PATH = "pneumonia_zip_mapping.csv"
OUT_DIR = "data/nih_raw/images"
os.makedirs(OUT_DIR, exist_ok=True)

HF_COMMIT = "d444bf39439f4fbfac6922e2472b1bf2554309e7"
HF_BASE = f"https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/{HF_COMMIT}/data/images"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ALL_ZIPS = [f"images_{i:03d}.zip" for i in range(1, 13)]

df = pd.read_csv(CSV_PATH)
needed_names = set(df["Image Index"].unique())
# don't re-fetch what's already on disk from the previous partial run
still_needed = {f for f in needed_names if not os.path.exists(os.path.join(OUT_DIR, f))}
print(f"{len(needed_names)} unique images needed, {len(still_needed)} not yet on disk.")

# --- Phase 1: build the REAL filename -> zip lookup ---
print("Indexing all 12 archives (central directories only, not full files) ...")
location = {}  # basename -> (zip_name, member_path_in_zip)
for zname in tqdm(ALL_ZIPS, desc="indexing"):
    url = f"{HF_BASE}/{zname}?download=true"
    try:
        with RemoteZip(url, headers=HEADERS) as rz:
            for member in rz.namelist():
                location[os.path.basename(member)] = (zname, member)
    except RangeNotSupported:
        raise RuntimeError(f"{zname}'s host doesn't support range requests -- can't index it remotely.")

found = {f: location[f] for f in still_needed if f in location}
truly_missing = sorted(still_needed - set(found))
print(f"{len(found)}/{len(still_needed)} remaining images located across the 12 archives.")
if truly_missing:
    print(f"{len(truly_missing)} image(s) are NOT present in ANY of the 12 archives -- "
          f"these need separate investigation (typo in filename? removed in a later "
          f"dataset revision? check against Data_Entry_2017_v2020.csv). First few: "
          f"{truly_missing[:10]}")
    with open("missing_images_report.txt", "w") as f:
        f.write("\n".join(truly_missing))
    print("Full list written to missing_images_report.txt")

# --- Phase 2: fetch only the bytes needed, from wherever they really live ---
by_zip = {}
for fname, (zname, member) in found.items():
    by_zip.setdefault(zname, []).append((fname, member))

total_bytes = 0
for zname, items in by_zip.items():
    url = f"{HF_BASE}/{zname}?download=true"
    with RemoteZip(url, headers=HEADERS) as rz:
        for fname, member in tqdm(items, desc=f"downloading from {zname}"):
            dest = os.path.join(OUT_DIR, fname)
            try:
                data = rz.read(member)
            except Exception as e:
                print(f"  FAILED to fetch {fname}: {e}")
                continue
            with open(dest, "wb") as out:
                out.write(data)
            total_bytes += len(data)

now_have = sum(1 for f in needed_names if os.path.exists(os.path.join(OUT_DIR, f)))
print(f"~{total_bytes / 1e6:.1f} MB transferred this run.")
print(f"Now have {now_have}/{len(needed_names)} of the needed images on disk.")
if now_have < len(needed_names):
    print(f"Still short {len(needed_names) - now_have} -- see missing_images_report.txt")