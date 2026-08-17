"""Stratified k-fold CV — v3. Adds exact-duplicate removal (MD5) before
folding — including files whose bytes appear under BOTH classes (known
Kermany label-conflict issue) — and per-class recall per fold so NORMAL
failures can never hide inside an accuracy mean again."""
import argparse
import glob
import hashlib
import json
import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import losses
from model_architecture import build_model, unfreeze_top_layers, get_preprocess_fn
from train import compute_focal_alpha


def build_file_dataframe():
    rows = []
    for class_name in config.CLASS_NAMES:
        for path in glob.glob(os.path.join(config.TRAIN_DIR, class_name, "*")):
            rows.append({"filepath": path, "class": class_name})
    return pd.DataFrame(rows)


def dedupe_dataframe(df):
    groups = {}
    for idx, row in df.iterrows():
        with open(row["filepath"], "rb") as f:
            groups.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(idx)
    keep, dup_same, conflict = [], 0, 0
    for idxs in groups.values():
        classes = {df.loc[i, "class"] for i in idxs}
        if len(classes) > 1:
            conflict += len(idxs)
        else:
            keep.append(idxs[0])
            dup_same += len(idxs) - 1
    print(f"Dedup: dropped {dup_same} same-class duplicates, {conflict} cross-class conflict files.")
    return df.loc[sorted(keep)].reset_index(drop=True)


def run_fold(activation, train_df, val_df, fold_idx, preprocess_fn):
    aug = dict(horizontal_flip=True, rotation_range=10, zoom_range=0.1,
               brightness_range=(0.85, 1.15)) if config.USE_AUGMENTATION else {}
    train_gen = ImageDataGenerator(preprocessing_function=preprocess_fn, **aug).flow_from_dataframe(
        train_df, x_col="filepath", y_col="class", target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE, class_mode="categorical",
        classes=config.CLASS_NAMES, shuffle=True, seed=config.RANDOM_SEED)
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_fn).flow_from_dataframe(
        val_df, x_col="filepath", y_col="class", target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE, class_mode="categorical",
        classes=config.CLASS_NAMES, shuffle=False)
    model = build_model(activation)
    focal_alpha = compute_focal_alpha(train_gen) if config.LOSS_FUNCTION == "focal" else None
    loss_fn = losses.get_loss(config.LOSS_FUNCTION, gamma=config.FOCAL_GAMMA, alpha=focal_alpha)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.WARMUP_LR), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=config.WARMUP_EPOCHS, verbose=1)
    unfreeze_top_layers(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.FINE_TUNE_LR), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=config.FINE_TUNE_EPOCHS,
              callbacks=[EarlyStopping(monitor=config.EARLY_STOPPING_MONITOR,
                                       patience=config.EARLY_STOPPING_PATIENCE,
                                       restore_best_weights=True, verbose=1)], verbose=1)
    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=0)
    y_true = val_gen.classes
    y_pred = np.argmax(y_prob, axis=1)
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "fold": fold_idx,
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "val_log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "val_recall_per_class": {n: float(r) for n, r in zip(config.CLASS_NAMES, recalls)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", choices=["sigmoid", "softmax"], required=True)
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    args = parser.parse_args()
    random.seed(config.RANDOM_SEED); np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    df = build_file_dataframe()
    if config.CV_DEDUPLICATE_EXACT:
        df = dedupe_dataframe(df)
    print(f"Training-pool images after dedup: {len(df)} ({df['class'].value_counts().to_dict()})")
    pf = get_preprocess_fn()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.RANDOM_SEED)
    fold_results = []
    start = time.time()
    for fold_idx, (tr, va) in enumerate(skf.split(df["filepath"], df["class"]), start=1):
        print(f"\nFold {fold_idx}/{args.folds} - {args.activation}")
        r = run_fold(args.activation, df.iloc[tr].reset_index(drop=True),
                     df.iloc[va].reset_index(drop=True), fold_idx, pf)
        fold_results.append(r)
        print(f"Fold {fold_idx}: {r}")
    accs = [r["val_accuracy"] for r in fold_results]
    f1s = [r["val_macro_f1"] for r in fold_results]
    lls = [r["val_log_loss"] for r in fold_results]
    nr = [r["val_recall_per_class"]["NORMAL"] for r in fold_results]
    summary = {"activation": args.activation, "folds": args.folds, "fold_results": fold_results,
               "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
               "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
               "log_loss_mean": float(np.mean(lls)), "log_loss_std": float(np.std(lls)),
               "normal_recall_mean": float(np.mean(nr)),
               "total_time_seconds": round(time.time() - start, 1)}
    with open(os.path.join(config.CV_DIR, f"cv_results_{args.activation}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{args.activation.upper()} CV: acc={summary['accuracy_mean']:.4f} "
          f"macroF1={summary['macro_f1_mean']:.4f} NORMAL_recall={summary['normal_recall_mean']:.4f}")


if __name__ == "__main__":
    main()