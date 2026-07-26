"""
Stratified k-fold cross-validation — an optional, more rigorous alternative
to a single train/val split. EXPENSIVE: trains config.CV_FOLDS full models
per activation, each with the same two-phase (warm-up + fine-tune) schedule
as train.py. Given CPU training times observed in this project (each single
model already takes multiple hours), a 5-fold run means ~5x that per
activation — budget accordingly, and ideally run this on a GPU.

This does NOT replace train.py / the primary train/test split — it's a
separate rigor check you run deliberately when you want a mean±std result
instead of a single number, e.g. for the paper's final reported numbers.
It only touches the train/ folder (never the held-out test/ folder, and
never the external NIH set) — those stay untouched as genuinely unseen data.

Run:
    python cross_validate.py --activation sigmoid --folds 5
    python cross_validate.py --activation softmax --folds 5
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config
import losses
from model_architecture import build_model, unfreeze_top_layers, get_preprocess_fn


def build_file_dataframe():
    """One row per training-set image: filepath + class label. Built from
    config.TRAIN_DIR's {NORMAL,PNEUMONIA} subfolders — the same pool
    train.py draws its 95/5 split from, just resplit K ways here instead."""
    rows = []
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        for path in glob.glob(os.path.join(class_dir, "*")):
            rows.append({"filepath": path, "class": class_name})
    return pd.DataFrame(rows)


def run_fold(activation, train_df, val_df, fold_idx, preprocess_fn):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        horizontal_flip=True, rotation_range=10, zoom_range=0.1, brightness_range=(0.85, 1.15),
    ) if config.USE_AUGMENTATION else ImageDataGenerator(preprocessing_function=preprocess_fn)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filepath", y_col="class", target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE, class_mode="categorical", classes=config.CLASS_NAMES,
        shuffle=True, seed=config.RANDOM_SEED,
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col="filepath", y_col="class", target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE, class_mode="categorical", classes=config.CLASS_NAMES,
        shuffle=False,
    )

    model = build_model(activation)
    loss_fn = losses.get_loss(config.LOSS_FUNCTION, gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.WARMUP_LR),
                  loss=loss_fn, metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=config.WARMUP_EPOCHS, verbose=1)

    unfreeze_top_layers(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LR),
                  loss=loss_fn, metrics=["accuracy"])
    model.fit(
        train_gen, validation_data=val_gen, epochs=config.FINE_TUNE_EPOCHS,
        callbacks=[EarlyStopping(monitor=config.EARLY_STOPPING_MONITOR,
                                  patience=config.EARLY_STOPPING_PATIENCE,
                                  restore_best_weights=True, verbose=1)],
        verbose=1,
    )

    val_gen.reset()
    y_prob = model.predict(val_gen, verbose=0)
    y_true = val_gen.classes
    y_pred = np.argmax(y_prob, axis=1)

    return {
        "fold": fold_idx,
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "val_log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", choices=["sigmoid", "softmax"], required=True)
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    args = parser.parse_args()

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    df = build_file_dataframe()
    print(f"Total training-pool images: {len(df)} ({df['class'].value_counts().to_dict()})")

    preprocess_fn = get_preprocess_fn()
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.RANDOM_SEED)

    fold_results = []
    start = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df["filepath"], df["class"]), start=1):
        print(f"\n{'='*70}\nFold {fold_idx}/{args.folds} — {args.activation}\n{'='*70}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        result = run_fold(args.activation, train_df, val_df, fold_idx, preprocess_fn)
        fold_results.append(result)
        print(f"Fold {fold_idx} result: {result}")

    elapsed = time.time() - start
    accs = [r["val_accuracy"] for r in fold_results]
    f1s = [r["val_macro_f1"] for r in fold_results]
    losses_ = [r["val_log_loss"] for r in fold_results]

    summary = {
        "activation": args.activation,
        "folds": args.folds,
        "fold_results": fold_results,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "log_loss_mean": float(np.mean(losses_)),
        "log_loss_std": float(np.std(losses_)),
        "total_time_seconds": round(elapsed, 1),
    }

    out_path = os.path.join(config.CV_DIR, f"cv_results_{args.activation}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.activation.upper()} — {args.folds}-fold CV summary:")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Macro-F1: {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"  Log loss: {summary['log_loss_mean']:.4f} +/- {summary['log_loss_std']:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
