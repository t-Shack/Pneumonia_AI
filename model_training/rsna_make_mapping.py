"""
Builds the RSNA download mapping.
1. Gets stage_2_detailed_class_info.csv (local copy, or from the mirror).
2. Keeps class == "Normal" -> NORMAL and "Lung Opacity" -> PNEUMONIA.
   Drops "No Lung Opacity / Not Normal" (the CSV states it explicitly).
3. Resolves each patientId's shard folder by listing the mirror once.
4. Stratified 85/15 train/test split, writes rsna_mapping.csv.
Run: python rsna_make_mapping.py
Needs: pip install huggingface_hub pandas requests
"""
import os
import requests
import pandas as pd

HF_REPO = "Baldezo313/rsna-pneumonia-dataset"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
CSV_NAME = "stage_2_detailed_class_info.csv"
RAW_DIR = os.path.join("data", "rsna_raw")
CLASS_MAP = {"Normal": "NORMAL", "Lung Opacity": "PNEUMONIA"}


def get_csv():
    os.makedirs(RAW_DIR, exist_ok=True)
    local = os.path.join(RAW_DIR, CSV_NAME)
    if not os.path.exists(local):
        print(f"Downloading {CSV_NAME} from mirror ...")
        r = requests.get(f"{HF_BASE}/{CSV_NAME}", timeout=120)
        r.raise_for_status()
        with open(local, "wb") as f:
            f.write(r.content)
    return local


def build_shard_index():
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        raise SystemExit("pip install huggingface_hub   (needed once, to list the mirror)")
    print("Listing mirror files (one-time) ...")
    index = {}
    for path in list_repo_files(HF_REPO, repo_type="dataset"):
        folder, _, name = path.partition("/")
        if folder.startswith("stage_2_train_images") and name.endswith(".dcm"):
            index[name[: -len(".dcm")]] = folder
    return index


def main():
    df = pd.read_csv(get_csv())
    print(f"CSV rows: {len(df)}")
    print(f"class values: {df['class'].value_counts().to_dict()}")
    df = df[df["class"].isin(CLASS_MAP)].copy()
    df["label"] = df["class"].map(CLASS_MAP)

    index = build_shard_index()
    df["shard"] = df["patientId"].map(index)
    missing = int(df["shard"].isna().sum())
    if missing:
        print(f"WARNING: {missing} patientIds not found in any mirror shard - dropped.")
        df = df.dropna(subset=["shard"])

    # stratified holdout so DATA_DIR/train + DATA_DIR/test both exist
    from sklearn.model_selection import train_test_split
    df["split"] = "train"
    _, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42)
    df.loc[df["patientId"].isin(test_df["patientId"]), "split"] = "test"

    df[["patientId", "class", "label", "shard", "split"]].to_csv("rsna_mapping.csv", index=False)
    for label in ("NORMAL", "PNEUMONIA"):
        sub = df[df["label"] == label]
        print(f"{label}: total={len(sub)} train={(sub.split == 'train').sum()} "
              f"test={(sub.split == 'test').sum()}")
    print("Wrote rsna_mapping.csv")


if __name__ == "__main__":
    main()