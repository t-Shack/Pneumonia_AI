"""
Builds the external test set from a downloaded NIH ChestX-ray14 sample:
filters Data_Entry_2017.csv for Pneumonia and No-Finding cases, locates the
matching image files (searched recursively), and copies them into
config.EXTERNAL_TEST_DIR/{NORMAL,PNEUMONIA}/.

Labeling rule matches CheXNet's convention: PNEUMONIA-positive if
"Pneumonia" appears anywhere in Finding Labels; NORMAL only if Finding
Labels is exactly "No Finding".

Run:
    python external_dataset_prepare.py [--balance-ratio 1.0] [--max-pneumonia N]
"""

import argparse
import os
import random
import shutil

import pandas as pd

import config


def build_filename_index(root_dir):
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                index[fname] = os.path.join(dirpath, fname)
    return index


def filter_labels(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    pneumonia_mask = df["Finding Labels"].str.contains("Pneumonia", case=False, na=False)
    normal_mask = df["Finding Labels"].str.strip().str.lower() == "no finding"

    pneumonia_df = df[pneumonia_mask]
    normal_df = df[normal_mask]

    print(f"CSV rows: {len(df)}")
    print(f"  Pneumonia-labeled: {len(pneumonia_df)}")
    print(f"  No Finding (normal): {len(normal_df)}")

    if len(normal_df) == 0:
        print("  *** WARNING: 0 No-Finding rows matched in the CSV itself. "
              "Check the CSV wasn't truncated/filtered before this step. ***")

    return pneumonia_df, normal_df


def copy_matches(df, filename_index, dest_dir, label, max_count=None, seed=42):
    filenames = df["Image Index"].tolist()
    if max_count is not None and len(filenames) > max_count:
        random.Random(seed).shuffle(filenames)
        filenames = filenames[:max_count]

    os.makedirs(dest_dir, exist_ok=True)
    copied, missing = 0, 0
    for fname in filenames:
        src = filename_index.get(fname)
        if src is None:
            missing += 1
            continue
        shutil.copy2(src, os.path.join(dest_dir, fname))
        copied += 1

    print(f"{label}: copied {copied}, missing (not found under NIH_RAW_IMAGES_DIR) {missing}")
    if copied == 0 and len(filenames) > 0:
        print(f"  *** {label}: 0 files copied despite {len(filenames)} CSV matches — none of those "
              f"filenames were found under {config.NIH_RAW_IMAGES_DIR}. You likely only downloaded "
              f"images for the OTHER class. Download some {label} images too. ***")
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance-ratio", type=float, default=1.0,
                         help="NORMAL:PNEUMONIA count ratio in the output set (default 1.0 = balanced).")
    parser.add_argument("--max-pneumonia", type=int, default=None,
                         help="Optional cap on Pneumonia images used (default: use all matches).")
    args = parser.parse_args()

    if not os.path.exists(config.NIH_CSV_PATH):
        print(f"CSV not found at {config.NIH_CSV_PATH} — see README.md 'External dataset setup'.")
        return
    if not os.path.isdir(config.NIH_RAW_IMAGES_DIR):
        print(f"Image folder not found at {config.NIH_RAW_IMAGES_DIR} — see README.md.")
        return

    pneumonia_df, normal_df = filter_labels(config.NIH_CSV_PATH)

    print("\nIndexing downloaded image files (recursive search)...")
    filename_index = build_filename_index(config.NIH_RAW_IMAGES_DIR)
    print(f"Found {len(filename_index)} PNG files under {config.NIH_RAW_IMAGES_DIR}")

    n_pneumonia_target = args.max_pneumonia or len(pneumonia_df)
    n_normal_target = int(n_pneumonia_target * args.balance_ratio)
    print(f"\nTarget: up to {n_pneumonia_target} PNEUMONIA, up to {n_normal_target} NORMAL")

    n_pneumonia = copy_matches(pneumonia_df, filename_index,
                                os.path.join(config.EXTERNAL_TEST_DIR, "PNEUMONIA"),
                                "PNEUMONIA", max_count=n_pneumonia_target, seed=config.RANDOM_SEED)
    n_normal = copy_matches(normal_df, filename_index,
                             os.path.join(config.EXTERNAL_TEST_DIR, "NORMAL"),
                             "NORMAL", max_count=n_normal_target, seed=config.RANDOM_SEED)

    print(f"\nExternal test set ready at {config.EXTERNAL_TEST_DIR}: {n_normal} NORMAL, {n_pneumonia} PNEUMONIA.")
    if n_normal == 0 or n_pneumonia == 0:
        print("*** At least one class has ZERO images. evaluate_external.py will run but the "
              "accuracy/precision/recall numbers for the missing class will be meaningless. "
              "Fix the download before trusting any external-set results. ***")
    print("\nNext: python evaluate_external.py")


if __name__ == "__main__":
    main()
