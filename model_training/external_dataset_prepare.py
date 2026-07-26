"""
Builds the external test set from a downloaded NIH ChestX-ray14 sample:
filters Data_Entry_2017.csv for Pneumonia and No-Finding cases, locates the
matching image files (searched recursively, so it doesn't matter whether
you downloaded the images_001..images_012 folder structure or a flat one),
and copies them into config.EXTERNAL_TEST_DIR/{NORMAL,PNEUMONIA}/.

Labeling rule matches CheXNet's own convention: an image counts as
PNEUMONIA-positive if "Pneumonia" appears anywhere in its Finding Labels
(co-occurring findings allowed), and NORMAL only if Finding Labels is
exactly "No Finding". This keeps the external set comparable to the
literature's reference point rather than inventing a stricter rule.

Prerequisites (see README.md "External dataset setup"):
  1. Download Data_Entry_2017.csv -> config.NIH_CSV_PATH
  2. Download the NIH image files you need -> anywhere under config.NIH_RAW_IMAGES_DIR
     (tip: filter the CSV first, then use `kaggle datasets download -f <filename>`
     per image instead of pulling the full ~42GB archive)

Run:
    python external_dataset_prepare.py [--balance-ratio 1.0] [--max-per-class N]
"""

import argparse
import os
import random
import shutil

import pandas as pd

import config


def build_filename_index(root_dir):
    """filename -> full path, searched recursively so any NIH folder layout
    (images_001/images/*.png, or a flat images/, etc.) works."""
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                index[fname] = os.path.join(dirpath, fname)
    return index


def filter_labels(csv_path):
    df = pd.read_csv(csv_path)
    # Column is usually "Image Index" / "Finding Labels" / "Patient ID" —
    # NIH has used slightly different header casing across CSV releases.
    df.columns = [c.strip() for c in df.columns]

    pneumonia_mask = df["Finding Labels"].str.contains("Pneumonia", case=False, na=False)
    normal_mask = df["Finding Labels"].str.strip().str.lower() == "no finding"

    pneumonia_df = df[pneumonia_mask]
    normal_df = df[normal_mask]

    print(f"CSV rows: {len(df)}")
    print(f"  Pneumonia-labeled (any co-occurring finding allowed): {len(pneumonia_df)}")
    print(f"    unique patients: {pneumonia_df['Patient ID'].nunique() if 'Patient ID' in df.columns else 'n/a'}")
    print(f"  No Finding (normal): {len(normal_df)}")
    print(f"    unique patients: {normal_df['Patient ID'].nunique() if 'Patient ID' in df.columns else 'n/a'}")

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
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balance-ratio", type=float, default=1.0,
        help="NORMAL:PNEUMONIA count ratio in the output set (default 1.0 = balanced). "
             "Pneumonia is the rare class in NIH ChestX-ray14, so this caps how many "
             "No-Finding images get pulled in, not how many Pneumonia images are used.",
    )
    parser.add_argument(
        "--max-pneumonia", type=int, default=None,
        help="Optional cap on Pneumonia images used (default: use all matches).",
    )
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

    n_pneumonia = copy_matches(
        pneumonia_df, filename_index,
        os.path.join(config.EXTERNAL_TEST_DIR, "PNEUMONIA"),
        "PNEUMONIA", max_count=n_pneumonia_target, seed=config.RANDOM_SEED,
    )
    n_normal = copy_matches(
        normal_df, filename_index,
        os.path.join(config.EXTERNAL_TEST_DIR, "NORMAL"),
        "NORMAL", max_count=n_normal_target, seed=config.RANDOM_SEED,
    )

    print(f"\nExternal test set ready at {config.EXTERNAL_TEST_DIR}: "
          f"{n_normal} NORMAL, {n_pneumonia} PNEUMONIA.")
    if n_pneumonia < (args.max_pneumonia or len(pneumonia_df)) * 0.5:
        print("WARNING: fewer than half the expected Pneumonia images were found — "
              "you likely haven't downloaded enough of the NIH image archive yet. "
              "Check which filenames are missing and download those specifically.")
    print("\nNext: python evaluate_external.py")


if __name__ == "__main__":
    main()
