"""
Trains sigmoid and softmax variants on top of a transfer-learned backbone
(DenseNet121 by default), using a standard two-phase schedule:
  Phase 1 (warm-up):  backbone frozen, train the new head only, higher LR.
  Phase 2 (fine-tune): top fraction of backbone unfrozen, whole model
                        trained together at a much lower LR, with early
                        stopping.

Run:
    python train.py
"""

import json
import os
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

import config
import losses
from data_pipeline import get_generators, print_dataset_summary
from model_architecture import build_model, unfreeze_top_layers


def compute_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    weight_dict = {int(i): float(w) for i, w in zip(np.unique(labels), weights)}
    print(f"Class weights ({train_gen.class_indices}): {weight_dict}")
    return weight_dict


def train_one(activation: str, train_gen, val_gen, class_weight=None):
    print(f"\n{'='*70}\nTraining {activation.upper()} — {config.BACKBONE}\n{'='*70}")

    model = build_model(activation)
    loss_fn = losses.get_loss(config.LOSS_FUNCTION, gamma=config.FOCAL_GAMMA, alpha=config.FOCAL_ALPHA)

    history_csv_path = os.path.join(config.HISTORY_DIR, f"history_{activation}.csv")
    callbacks = [
        EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR, patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        CSVLogger(history_csv_path),
    ]

    # --- Phase 1: warm-up, frozen backbone -----------------------------------
    print(f"\n--- Phase 1: warm-up ({config.WARMUP_EPOCHS} epochs, backbone frozen) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.WARMUP_LR),
        loss=loss_fn, metrics=config.METRICS,
    )
    start = time.time()
    warmup_history = model.fit(
        train_gen, validation_data=val_gen, epochs=config.WARMUP_EPOCHS,
        class_weight=class_weight, verbose=1,
    )

    # --- Phase 2: fine-tune, top fraction of backbone unfrozen ---------------
    print(f"\n--- Phase 2: fine-tune (up to {config.FINE_TUNE_EPOCHS} epochs, "
          f"top {config.FINE_TUNE_UNFREEZE_FRACTION:.0%} of backbone unfrozen) ---")
    unfreeze_top_layers(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LR),
        loss=loss_fn, metrics=config.METRICS,
    )
    fine_tune_history = model.fit(
        train_gen, validation_data=val_gen, epochs=config.FINE_TUNE_EPOCHS,
        callbacks=callbacks, class_weight=class_weight, verbose=1,
    )
    elapsed = time.time() - start

    # Merge warm-up + fine-tune history into one continuous record for the charts.
    merged_history = {}
    for key in warmup_history.history:
        merged_history[key] = warmup_history.history[key] + fine_tune_history.history.get(key, [])
    phase_boundary_epoch = len(warmup_history.history["loss"])  # for annotating charts later

    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{activation}.{config.MODEL_FORMAT}")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    with open(os.path.join(config.HISTORY_DIR, f"history_{activation}.json"), "w") as f:
        json.dump({"history": merged_history, "phase_boundary_epoch": phase_boundary_epoch}, f, indent=2)

    return {
        "label": activation,
        "activation": activation,
        "backbone": config.BACKBONE,
        "model_path": model_path,
        "warmup_epochs": len(warmup_history.history["loss"]),
        "fine_tune_epochs": len(fine_tune_history.history["loss"]),
        "training_time_seconds": round(elapsed, 1),
        "best_val_loss": min(merged_history["val_loss"]),
        "best_val_accuracy": max(merged_history["val_accuracy"]),
        "loss_function": config.LOSS_FUNCTION,
        "class_weight_applied": class_weight,
    }


def main():
    # Seed every RNG source for reproducibility across runs (see README —
    # this is what fixed the sigmoid/softmax "winner" flipping between runs
    # in the earlier from-scratch pipeline).
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    train_gen, val_gen, test_gen = get_generators()
    print_dataset_summary(train_gen, val_gen, test_gen)

    deployment_config = {
        "img_size": list(config.IMG_SIZE),
        "preprocessing": config.BACKBONE,   # webapp/inference.py looks this up
        "class_indices": train_gen.class_indices,
        "model_format": config.MODEL_FORMAT,
        "backbone": config.BACKBONE,
    }
    with open(os.path.join(config.OUTPUT_DIR, "deployment_config.json"), "w") as f:
        json.dump(deployment_config, f, indent=2)

    class_weight = compute_class_weights(train_gen) if config.USE_CLASS_WEIGHTS else None

    results = []
    for activation in ("sigmoid", "softmax"):
        results.append(train_one(activation, train_gen, val_gen, class_weight=class_weight))

    with open(os.path.join(config.OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete. Summary:")
    for r in results:
        print(f"  {r['label']:8s} | warmup={r['warmup_epochs']:2d} + fine_tune={r['fine_tune_epochs']:2d} "
              f"| best_val_acc={r['best_val_accuracy']:.4f} | best_val_loss={r['best_val_loss']:.4f} "
              f"| time={r['training_time_seconds']}s")


if __name__ == "__main__":
    main()
