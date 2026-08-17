"""
nih_train_mix_mapping.py — builds NIH train-mix mapping CSVs, disjoint from
the external test set. Run: python nih_train_mix_mapping.py
"""
import os
import pandas as pd
from tqdm import tqdm
from remotezip import RemoteZip, RangeNotSupported

ENTRY_CSV = "data/nih_raw/Data_Entry_2017.csv"
PNEUM_OUT = "trainmix_pneumonia_zip_mapping.csv"
NORMAL_OUT = "trainmix_normal_zip_mapping.csv"
MISSING_REPORT = "trainmix_missing_report.txt"
HF_COMMIT = "d444bf39439f4fbfac6922e2472b1bf2554309e7"
HF_BASE = f"https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/{HF_COMMIT}/data/images"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ALL_ZIPS = [f"images_{i:03d}.zip" for i in range(1, 13)]
SEED = 7
N_NORMAL = 2000
N_PNEUM = 800
# Adult pneumonia-like proxy, used only because the external test consumed all
# true 'Pneumonia' images. Set [] for a normals-only mix.
PROXY_PNEUMONIA_LABELS = ["Consolidation"]

entries = pd.read_csv(ENTRY_CSV)
used = set()
for csv in ("pneumonia_zip_mapping.csv", "normal_zip_mapping.csv"):
    if os.path.exists(csv):
        used |= set(pd.read_csv(csv)["Image Index"])
print(f"Excluding {len(used)} external-test images.")

def draw(pool, n):
    return pool.sample(min(n, len(pool)), random_state=SEED) if len(pool) else pool

# --- pneumonia pool ---
pneum_pool = entries[entries["Finding Labels"].str.contains("Pneumonia", regex=False)]
pneum_pool = pneum_pool[~pneum_pool["Image Index"].isin(used)]
print(f"{len(pneum_pool)} unused true-'Pneumonia' images available.")
if len(pneum_pool) < N_PNEUM and PROXY_PNEUMONIA_LABELS:
    mask = pd.Series(False, index=entries.index)
    for lab in PROXY_PNEUMONIA_LABELS:
        mask |= entries["Finding Labels"].str.contains(lab, regex=False)
    proxy = entries[mask]
    proxy = proxy[~proxy["Image Index"].isin(used | set(pneum_pool["Image Index"]))]
    print(f"True-'Pneumonia' pool exhausted -> adding {len(proxy)} adult "
          f"{'/'.join(PROXY_PNEUMONIA_LABELS)} images as pneumonia-like examples.")
    pneum_pool = pd.concat([pneum_pool, proxy])
pneum = draw(pneum_pool, N_PNEUM)

# --- normal pool ---
norm_pool = entries[entries["Finding Labels"] == "No Finding"]
norm_pool = norm_pool[~norm_pool["Image Index"].isin(used)]
print(f"{len(norm_pool)} unused 'No Finding' images available.")
norm = draw(norm_pool, N_NORMAL)
print(f"Train-mix sample: {len(pneum)} pneumonia-like, {len(norm)} normal.")

# --- resolve Zip_File against the real archives ---
location = {}
for zname in tqdm(ALL_ZIPS, desc="indexing"):
    try:
        with RemoteZip(f"{HF_BASE}/{zname}?download=true", headers=HEADERS) as rz:
            for member in rz.namelist():
                location[os.path.basename(member)] = zname
    except RangeNotSupported:
        raise RuntimeError(f"{zname}'s host doesn't support range requests.")

all_missing = []
for df, out, label in ((pneum, PNEUM_OUT, "pneumonia"), (norm, NORMAL_OUT, "normal")):
    df = df[["Image Index", "Finding Labels"]].copy()
    df["Zip_File"] = df["Image Index"].map(location)
    all_missing += df.loc[df["Zip_File"].isna(), "Image Index"].tolist()
    df = df.dropna(subset=["Zip_File"])
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} {label} images to {out}")
if all_missing:
    with open(MISSING_REPORT, "w") as f:
        f.write("\n".join(all_missing))
    print(f"{len(all_missing)} image(s) not found in any archive - see {MISSING_REPORT}")