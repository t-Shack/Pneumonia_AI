"""
Evaluates every trained model against the EXTERNAL test set (NIH
ChestX-ray14, prepared by external_dataset_prepare.py) — the actual
cross-institution generalization check. Never used for training.

Run after external_dataset_prepare.py has built config.EXTERNAL_TEST_DIR:
    python evaluate_external.py
"""

import json
import os

import config
from data_pipeline import get_external_test_generator
from evaluate import discover_models, evaluate_model


def main():
    if not os.path.isdir(config.EXTERNAL_TEST_DIR):
        print(f"{config.EXTERNAL_TEST_DIR} not found. Run external_dataset_prepare.py first.")
        return

    test_gen = get_external_test_generator()
    print(f"External test set: {test_gen.samples} images "
          f"({test_gen.class_indices})")

    models = discover_models()
    if not models:
        print(f"No pneumonia_*.keras models found in {config.MODELS_DIR}")
        return

    all_results = {}
    for label, model_path in models.items():
        all_results[label] = evaluate_model(model_path, test_gen, label)

    out_path = os.path.join(config.METRICS_DIR, "external_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved external evaluation metrics to {out_path}")

    print("\nPrimary test set vs. external (NIH) test set — compare these "
          "against outputs/metrics/test_evaluation.json by hand, or just "
          "look at generate_charts.py's 10_external_comparison.png once "
          "you've run both evaluate.py and this script.")
    for label, r in all_results.items():
        print(f"\n{label.upper()} (external)")
        print(f"  Test accuracy: {r['test_accuracy']:.4f}")
        print(f"  Test loss:     {r['test_loss']:.4f}")
        print(f"  ROC-AUC:       {r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
