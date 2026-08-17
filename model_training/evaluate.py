"""Primary (Kermany) test evaluation — v3.
Adds balanced accuracy + explicit per-class recall to every result, and an
'as_deployed' block (metrics at the deployed threshold) when the config names
this model. argmax numbers remain for model-vs-model comparison only."""
import json
import os
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             roc_auc_score, precision_recall_curve)
from tensorflow.keras.models import load_model
import config
import losses  # noqa: F401
from data_pipeline import get_generators


def discover_models():
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
    pneu = config.CLASS_NAMES.index("PNEUMONIA")
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, pneu])
    prec, rec, _ = precision_recall_curve(y_true, y_prob[:, pneu])
    result = {
        "label": label,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "balanced_accuracy": float(np.mean([report[c]["recall"] for c in config.CLASS_NAMES])),
        "normal_recall": float(report["NORMAL"]["recall"]),
        "pneumonia_recall": float(report["PNEUMONIA"]["recall"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
        "roc_auc": float(roc_auc_score(y_true, y_prob[:, pneu])),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
    }
    dep_path = os.path.join(config.OUTPUT_DIR, "deployment_config.json")
    if os.path.exists(dep_path):
        with open(dep_path) as f:
            dep = json.load(f)
        if dep.get("best_model") == label and "decision_threshold" in dep:
            t = dep["decision_threshold"]
            y_pred_t = (y_prob[:, pneu] >= t).astype(int)
            rep_t = classification_report(y_true, y_pred_t, target_names=config.CLASS_NAMES, output_dict=True)
            result["as_deployed"] = {
                "decision_threshold": t,
                "accuracy": float(rep_t["accuracy"]),
                "normal_recall": float(rep_t["NORMAL"]["recall"]),
                "pneumonia_recall": float(rep_t["PNEUMONIA"]["recall"]),
                "confusion_matrix": confusion_matrix(y_true, y_pred_t).tolist(),
            }
    return result


def main():
    _, _, test_gen = get_generators()
    models = discover_models()
    if not models:
        print(f"No pneumonia_*.keras models found in {config.MODELS_DIR}")
        return
    all_results = {label: evaluate_model(path, test_gen, label) for label, path in models.items()}
    with open(os.path.join(config.METRICS_DIR, "test_evaluation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    for label, r in all_results.items():
        print(f"\n{label.upper()}")
        print(f"  Accuracy: {r['test_accuracy']:.4f} | Balanced: {r['balanced_accuracy']:.4f}")
        print(f"  NORMAL recall: {r['normal_recall']:.3f} | PNEUMONIA recall: {r['pneumonia_recall']:.3f}")
        print(f"  ROC-AUC: {r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()