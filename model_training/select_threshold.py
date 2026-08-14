"""
Picks a decision threshold for the PNEUMONIA probability, calibrated on the
VALIDATION set — never the test set, since choosing a threshold from test
data and then reporting test accuracy at that threshold would quietly bias
the very number being reported as "final" results.

The default 0.5/argmax cutoff was shown (both in offline evaluation and in
real usage — 3 real normal X-rays all misclassified as pneumonia) to favor
PNEUMONIA heavily: NORMAL recall was only ~61% despite a strong ROC-AUC
(0.94-0.97), meaning the model ranks cases well but the operating point was
badly placed. This script finds a better one (Youden's J: the threshold
that maximizes tpr - fpr) and writes it to deployment_config.json as
"decision_threshold" — that's what the webapp uses instead of naive argmax.

Also re-evaluates the winning model on the (untouched-for-threshold-
selection) test set AT that threshold, so you have an honest "as-deployed"
headline number, distinct from the argmax-based comparison numbers in
test_evaluation.json (which exist to compare models against each other, not
to describe final deployed performance).

Run after select_best_model.py:
    python select_threshold.py
"""

import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from tensorflow.keras.models import load_model

import config
import losses  # noqa: F401 — needed for load_model to deserialize focal loss
from data_pipeline import get_generators, get_validation_eval_generator


def find_threshold(y_true, y_prob_pneumonia, method="youdens_j"):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pneumonia)
    if method == "youdens_j":
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
    else:
        raise ValueError(f"Unknown THRESHOLD_SELECTION_METHOD: {method!r}")
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(1 - fpr[best_idx])


def sanity_check_threshold(threshold, normal_recall, pneumonia_recall):
    """Cheap insurance against exactly the failure mode that shipped a
    ~0.97 threshold earlier: Youden's J found a spurious spike on a small,
    misaligned validation sample rather than a genuine balance point — you
    could tell because BOTH recalls were mediocre (61%/46%) even on the
    data the threshold was chosen from, which a real balanced point
    shouldn't produce. Flags (doesn't silently accept) anything that looks
    like the same pattern."""
    warnings = []
    if threshold < 0.15 or threshold > 0.85:
        warnings.append(
            f"Threshold {threshold:.3f} is extreme (outside [0.15, 0.85]) — unusual for a "
            f"genuinely balanced operating point. Double-check the validation predictions/labels "
            f"are correctly aligned before trusting this."
        )
    if normal_recall < 0.5 or pneumonia_recall < 0.5:
        warnings.append(
            f"NORMAL recall={normal_recall:.1%} and PNEUMONIA recall={pneumonia_recall:.1%} at this "
            f"threshold — a real balance point shouldn't leave BOTH classes doing this poorly on the "
            f"very data the threshold was selected from. Likely a data/alignment problem, not a "
            f"genuine best-available tradeoff."
        )
    return warnings


def evaluate_at_threshold(y_true, y_prob_pneumonia, threshold):
    y_pred = (y_prob_pneumonia >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)
    return cm, report


def main():
    deployment_config_path = os.path.join(config.OUTPUT_DIR, "deployment_config.json")
    if not os.path.exists(deployment_config_path):
        print(f"{deployment_config_path} not found — run train.py and select_best_model.py first.")
        return
    with open(deployment_config_path) as f:
        deployment_config = json.load(f)

    label = deployment_config.get("best_model")
    if not label:
        print("deployment_config.json has no \"best_model\" field — run select_best_model.py first.")
        return

    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{label}.keras")
    if not os.path.exists(model_path):
        print(f"{model_path} not found.")
        return

    print(f"Calibrating decision threshold for the winning model: {label}")
    model = load_model(model_path)

    _, _, test_gen = get_generators()
    val_gen = get_validation_eval_generator()
    class_indices = val_gen.class_indices
    pneumonia_idx = class_indices["PNEUMONIA"]

    # Threshold chosen on VALIDATION data only (non-shuffled generator —
    # see get_validation_eval_generator()'s docstring for why that matters).
    val_gen.reset()
    y_val_prob = model.predict(val_gen, verbose=0)
    y_val_true = val_gen.classes
    threshold, val_pneumonia_recall, val_normal_recall = find_threshold(
        y_val_true, y_val_prob[:, pneumonia_idx], method=config.THRESHOLD_SELECTION_METHOD
    )
    print(f"\nSelected threshold ({config.THRESHOLD_SELECTION_METHOD}, on validation data): {threshold:.4f}")
    print(f"  Validation NORMAL recall at this threshold:    {val_normal_recall:.1%}")
    print(f"  Validation PNEUMONIA recall at this threshold: {val_pneumonia_recall:.1%}")

    warnings = sanity_check_threshold(threshold, val_normal_recall, val_pneumonia_recall)
    if warnings:
        print("\n*** SANITY CHECK FAILED — refusing to write this threshold automatically ***")
        for w in warnings:
            print(f"  - {w}")
        print("\nNot updating deployment_config.json. Investigate before rerunning — if you're "
              "confident the threshold is actually fine despite this warning, you can bypass this "
              "check by editing deployment_config.json's \"decision_threshold\" by hand.")
        return

    # Report on the test set — informational only, NOT used to pick the threshold.
    test_gen.reset()
    y_test_prob = model.predict(test_gen, verbose=0)
    y_test_true = test_gen.classes
    cm, report = evaluate_at_threshold(y_test_true, y_test_prob[:, pneumonia_idx], threshold)

    print(f"\nTest-set performance AT this threshold (the honest, as-deployed numbers):")
    print(f"  NORMAL:    precision={report['NORMAL']['precision']:.3f}  recall={report['NORMAL']['recall']:.3f}")
    print(f"  PNEUMONIA: precision={report['PNEUMONIA']['precision']:.3f}  recall={report['PNEUMONIA']['recall']:.3f}")
    print(f"  Overall accuracy: {report['accuracy']:.3f}")
    print(f"  Confusion matrix: {cm.tolist()}")

    deployment_config["decision_threshold"] = threshold
    with open(deployment_config_path, "w") as f:
        json.dump(deployment_config, f, indent=2)
    print(f"\nWrote \"decision_threshold\": {threshold:.4f} to {deployment_config_path}")
    print("Copy the updated deployment_config.json into webapp/models/ to deploy this calibration.")

    out_path = os.path.join(config.METRICS_DIR, "final_evaluation.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": label,
            "decision_threshold": threshold,
            "selection_method": config.THRESHOLD_SELECTION_METHOD,
            "selected_on": "validation_set",
            "validation_metrics_at_threshold": {
                "normal_recall": val_normal_recall,
                "pneumonia_recall": val_pneumonia_recall,
            },
            "test_confusion_matrix": cm.tolist(),
            "test_classification_report": report,
        }, f, indent=2)
    print(f"Full report saved to {out_path} — cite these as your final, as-deployed results "
          f"(not the argmax-based numbers in test_evaluation.json, which exist for model comparison only).")


if __name__ == "__main__":
    main()
