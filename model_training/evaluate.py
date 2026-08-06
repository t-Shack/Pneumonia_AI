"""
Evaluates every trained model against the primary (Kermany) test set.

Run after train.py:
    python evaluate.py
"""

import json
import os

import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve,
)
from tensorflow.keras.models import load_model

import config
import losses  # noqa: F401 — needed for load_model to deserialize the custom focal loss
from data_pipeline import get_generators


def discover_models():
    """Finds every pneumonia_<label>.keras in MODELS_DIR."""
    models = {}
    if not os.path.isdir(config.MODELS_DIR):
        return models
    for fname in sorted(os.listdir(config.MODELS_DIR)):
        if fname.startswith("pneumonia_") and fname.endswith(".keras"):
            label = fname[len("pneumonia_"):-len(".keras")]
            models[label] = os.path.join(config.MODELS_DIR, fname)
    return models


def evaluate_model(model_path, test_gen, label):
    print(f"\nEvaluating {label} model...")
    model = load_model(model_path)

    test_gen.reset()
    loss, acc = model.evaluate(test_gen, verbose=0)

    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=0)
    y_true = test_gen.classes
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)

    pneumonia_idx = config.CLASS_NAMES.index("PNEUMONIA")
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, pneumonia_idx])
    auc = roc_auc_score(y_true, y_prob[:, pneumonia_idx])
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, pneumonia_idx])

    return {
        "label": label,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_auc": float(auc),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
    }


def main():
    _, _, test_gen = get_generators()

    models = discover_models()
    if not models:
        print(f"No pneumonia_*.keras models found in {config.MODELS_DIR}")
        return

    all_results = {}
    for label, model_path in models.items():
        all_results[label] = evaluate_model(model_path, test_gen, label)

    out_path = os.path.join(config.METRICS_DIR, "test_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved evaluation metrics to {out_path}")

    for label, r in all_results.items():
        print(f"\n{label.upper()}")
        print(f"  Test accuracy: {r['test_accuracy']:.4f}")
        print(f"  Test loss:     {r['test_loss']:.4f}")
        print(f"  ROC-AUC:       {r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
