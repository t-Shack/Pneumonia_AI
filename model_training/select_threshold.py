"""Decision-threshold calibration — v3.
Default: RECALL-CONSTRAINED — among thresholds keeping PNEUMONIA recall >=
config.THRESHOLD_MIN_PNEUMONIA_RECALL on validation, pick the lowest-FPR one
(highest threshold meeting the floor). The old unconstrained Youden-on-60-
normals is what shipped 0.387 and flagged ~40% of normals as pneumonia.
Chosen on validation only; test set reported AT the threshold (as-deployed)."""
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import config
import losses  # noqa: F401
from data_pipeline import get_generators, get_validation_eval_generator


def find_threshold(y_true, y_prob_pneumonia, method):
    y_true = np.asarray(y_true)
    y = np.asarray(y_prob_pneumonia)
    pos, neg = y[y_true == 1], y[y_true == 0]
    candidates = np.sort(np.unique(y))[::-1]  # descending thresholds
    recall = np.array([(pos >= t).mean() for t in candidates])
    fpr = np.array([(neg >= t).mean() for t in candidates])
    if method == "youdens_j":
        idx = int(np.argmax(recall - fpr))
    elif method == "recall_constrained":
        feasible = np.where(recall >= config.THRESHOLD_MIN_PNEUMONIA_RECALL)[0]
        if feasible.size == 0:
            print("WARNING: recall floor unattainable on validation; using max-recall threshold.")
            idx = int(np.argmax(recall))
        else:
            best = fpr[feasible].min()
            tie = feasible[fpr[feasible] == best]
            idx = int(tie[0])  # candidates descending -> first tie = highest threshold
    else:
        raise ValueError(f"Unknown THRESHOLD_SELECTION_METHOD: {method!r}")
    return float(candidates[idx]), float(recall[idx]), float(1 - fpr[idx])


def sanity_check_threshold(threshold, normal_recall, pneumonia_recall):
    warnings = []
    if threshold < 0.15 or threshold > 0.85:
        warnings.append(f"Threshold {threshold:.3f} outside [0.15, 0.85] - check label/prob alignment.")
    if normal_recall < 0.5 or pneumonia_recall < 0.5:
        warnings.append(f"Both recalls mediocre at selection time ({normal_recall:.1%}/{pneumonia_recall:.1%}) - likely data/alignment problem.")
    return warnings


def main():
    dep_path = os.path.join(config.OUTPUT_DIR, "deployment_config.json")
    if not os.path.exists(dep_path):
        print(f"{dep_path} not found - run train.py and select_best_model.py first.")
        return
    with open(dep_path) as f:
        deployment_config = json.load(f)
    label = deployment_config.get("best_model")
    if not label:
        print("deployment_config.json has no best_model - run select_best_model.py first.")
        return
    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{label}.keras")
    if not os.path.exists(model_path):
        print(f"{model_path} not found.")
        return
    print(f"Calibrating threshold for winning model: {label}")
    model = load_model(model_path)
    _, _, test_gen = get_generators()
    val_gen = get_validation_eval_generator()
    pneu_idx = val_gen.class_indices["PNEUMONIA"]

    val_gen.reset()
    y_val_prob = model.predict(val_gen, verbose=0)[:, pneu_idx]
    y_val_true = val_gen.classes
    threshold, val_pneu_recall, val_norm_recall = find_threshold(
        y_val_true, y_val_prob, method=config.THRESHOLD_SELECTION_METHOD)
    print(f"\nSelected threshold ({config.THRESHOLD_SELECTION_METHOD}, validation only): {threshold:.4f}")
    print(f"  Validation NORMAL recall:    {val_norm_recall:.1%}")
    print(f"  Validation PNEUMONIA recall: {val_pneu_recall:.1%}")
    warnings = sanity_check_threshold(threshold, val_norm_recall, val_pneu_recall)
    if warnings:
        print("\n*** SANITY CHECK FAILED - refusing to write threshold ***")
        for w in warnings:
            print("  -", w)
        return

    test_gen.reset()
    y_test_prob = model.predict(test_gen, verbose=0)[:, pneu_idx]
    y_test_true = test_gen.classes
    y_pred = (y_test_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test_true, y_pred)
    report = classification_report(y_test_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)
    print(f"\nTest-set performance AT this threshold (as-deployed numbers):")
    print(f"  NORMAL:    precision={report['NORMAL']['precision']:.3f} recall={report['NORMAL']['recall']:.3f}")
    print(f"  PNEUMONIA: precision={report['PNEUMONIA']['precision']:.3f} recall={report['PNEUMONIA']['recall']:.3f}")
    print(f"  Accuracy: {report['accuracy']:.3f} | CM: {cm.tolist()}")

    deployment_config["decision_threshold"] = threshold
    with open(dep_path, "w") as f:
        json.dump(deployment_config, f, indent=2)
    with open(os.path.join(config.METRICS_DIR, "final_evaluation.json"), "w") as f:
        json.dump({
            "model": label, "decision_threshold": threshold,
            "selection_method": config.THRESHOLD_SELECTION_METHOD,
            "recall_floor": config.THRESHOLD_MIN_PNEUMONIA_RECALL,
            "selected_on": "validation_set",
            "validation_metrics_at_threshold": {
                "normal_recall": val_norm_recall, "pneumonia_recall": val_pneu_recall},
            "test_confusion_matrix": cm.tolist(), "test_classification_report": report,
        }, f, indent=2)
    print(f"Wrote decision_threshold={threshold:.4f}; report in final_evaluation.json")


if __name__ == "__main__":
    main()