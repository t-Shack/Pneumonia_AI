"""
Generates all charts/figures. Works with any number of trained models.

Run after train.py, evaluate.py, robustness_test.py, and (optionally)
evaluate_external.py:
    python generate_charts.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model

import config
import losses  # noqa: F401
from data_pipeline import get_generators, get_raw_test_generator
from evaluate import discover_models
from model_architecture import get_preprocess_fn

sns.set_theme(style="whitegrid")
DPI = 200
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def _save(fig, name):
    path = os.path.join(config.CHARTS_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def discover_history_labels():
    pattern = os.path.join(config.HISTORY_DIR, "history_*.json")
    labels = [os.path.basename(p)[len("history_"):-len(".json")] for p in glob.glob(pattern)]
    priority = {"sigmoid": 0, "softmax": 1}
    return sorted(labels, key=lambda l: (priority.get(l, 2), l))


def ordered_labels(keys):
    priority = {"sigmoid": 0, "softmax": 1}
    return sorted(keys, key=lambda l: (priority.get(l, 2), l))


def load_history(label):
    raw = load_json(os.path.join(config.HISTORY_DIR, f"history_{label}.json"))
    if "history" in raw:
        return raw["history"], raw.get("phase_boundary_epoch")
    return raw, None


def chart_training_curves():
    labels = discover_history_labels()
    if not labels:
        print("No training history found — skipping.")
        return
    fig, axes = plt.subplots(2, len(labels), figsize=(6 * len(labels), 8), squeeze=False)
    for col, label in enumerate(labels):
        h, phase_boundary = load_history(label)
        epochs = range(1, len(h["loss"]) + 1)
        title = label.replace("_", " ").capitalize()

        ax_acc = axes[0][col]
        ax_acc.plot(epochs, h["accuracy"], label="Train")
        ax_acc.plot(epochs, h["val_accuracy"], label="Validation")
        if phase_boundary:
            ax_acc.axvline(phase_boundary + 0.5, color="gray", linestyle=":", linewidth=1)
        ax_acc.set_title(f"{title} model — Accuracy")
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy"); ax_acc.legend()

        ax_loss = axes[1][col]
        ax_loss.plot(epochs, h["loss"], label="Train")
        ax_loss.plot(epochs, h["val_loss"], label="Validation")
        if phase_boundary:
            ax_loss.axvline(phase_boundary + 0.5, color="gray", linestyle=":", linewidth=1)
        ax_loss.set_title(f"{title} model — Loss")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.legend()
    fig.tight_layout()
    _save(fig, "01_training_curves.png")


def chart_confusion_matrices(eval_results, filename="02_confusion_matrices.png", title_suffix=""):
    labels = ordered_labels(eval_results.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 5), squeeze=False)
    axes = axes[0]
    for ax, label in zip(axes, labels):
        cm = np.array(eval_results[label]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        ax.set_title(f"{label.replace('_', ' ').capitalize()}{title_suffix} — Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    _save(fig, filename)


def chart_roc_curves(eval_results, filename="03_roc_curves.png", title_suffix=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label in ordered_labels(eval_results.keys()):
        roc = eval_results[label]["roc_curve"]
        auc = eval_results[label]["roc_auc"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve{title_suffix} — Pneumonia Detection"); ax.legend()
    fig.tight_layout()
    _save(fig, filename)


def chart_pr_curves(eval_results):
    fig, ax = plt.subplots(figsize=(6, 6))
    for label in ordered_labels(eval_results.keys()):
        pr = eval_results[label]["pr_curve"]
        ax.plot(pr["recall"], pr["precision"], label=label)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Pneumonia Detection"); ax.legend()
    fig.tight_layout()
    _save(fig, "04_precision_recall_curves.png")


def chart_model_comparison(eval_results):
    labels = ordered_labels(eval_results.keys())
    accs = [eval_results[l]["test_accuracy"] for l in labels]
    losses_ = [eval_results[l]["test_loss"] for l in labels]
    colors = PALETTE[: len(labels)]
    fig, axes = plt.subplots(1, 2, figsize=(5 * len(labels), 4.5))
    axes[0].bar(labels, accs, color=colors); axes[0].set_title("Test Accuracy"); axes[0].set_ylim(0, 1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")
    axes[1].bar(labels, losses_, color=colors); axes[1].set_title("Test Loss")
    for i, v in enumerate(losses_):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    _save(fig, "05_model_comparison.png")


def chart_robustness(robustness_results):
    labels = ordered_labels(robustness_results.keys())
    degrade_names = list(next(iter(robustness_results.values()))["degradations"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, degrade_name in zip(axes, degrade_names):
        for color, label in zip(PALETTE, labels):
            model_results = robustness_results[label]
            baseline = model_results["baseline_accuracy"]
            levels = model_results["degradations"][degrade_name]
            level_names = list(levels.keys())
            accs = [levels[l]["accuracy"] for l in level_names]
            ax.plot(["clean"] + level_names, [baseline] + accs, marker="o", color=color, label=label)
        ax.set_title(degrade_name.replace("_", " ").title())
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle("Robustness Under Image Quality Degradation", y=1.02)
    fig.tight_layout()
    _save(fig, "06_robustness_degradation.png")


def chart_dataset_composition(train_gen, val_gen, test_gen):
    fig, ax = plt.subplots(figsize=(7, 5))
    splits = ["Train", "Validation", "Test"]
    generators = [train_gen, val_gen, test_gen]
    normal_counts, pneumonia_counts = [], []
    for gen in generators:
        classes = np.array(gen.classes)
        normal_idx = gen.class_indices["NORMAL"]
        pneumonia_idx = gen.class_indices["PNEUMONIA"]
        normal_counts.append(int((classes == normal_idx).sum()))
        pneumonia_counts.append(int((classes == pneumonia_idx).sum()))
    x = np.arange(len(splits)); width = 0.35
    ax.bar(x - width / 2, normal_counts, width, label="Normal")
    ax.bar(x + width / 2, pneumonia_counts, width, label="Pneumonia")
    ax.set_xticks(x); ax.set_xticklabels(splits); ax.set_ylabel("Number of images")
    ax.set_title("Dataset Class Distribution"); ax.legend()
    fig.tight_layout()
    _save(fig, "07_dataset_composition.png")


def chart_filter_progression():
    print("Skipping filter-progression chart — not applicable to a pretrained backbone.")


# add this function:
def chart_per_class_recall(eval_results, filename="08_per_class_recall.png", title_suffix=""):
    labels = ordered_labels(eval_results.keys())
    fig, ax = plt.subplots(figsize=(5 + len(labels), 5))
    x = np.arange(len(labels)); width = 0.35
    def recalls(r):
        rep = r["classification_report"]
        return rep["NORMAL"]["recall"], rep["PNEUMONIA"]["recall"]
    nr, pr = zip(*[recalls(eval_results[l]) for l in labels])
    ax.bar(x - width / 2, nr, width, label="NORMAL recall (specificity)")
    ax.bar(x + width / 2, pr, width, label="PNEUMONIA recall (sensitivity)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_title(f"Per-Class Recall{title_suffix}"); ax.legend()
    fig.tight_layout()
    _save(fig, filename)

# in main(), after chart_model_comparison(eval_results): add
#     chart_per_class_recall(eval_results)
# and inside the external block, after the 02b/03b charts: add
#     chart_per_class_recall(external_results, filename="08b_external_per_class_recall.png",
#                            title_suffix=" (External/NIH)")


def _pick_balanced_samples(test_gen, per_class):
    class_indices = test_gen.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    test_gen.reset()
    images_by_class = {name: [] for name in class_indices}
    seen_batches, max_batches = 0, len(test_gen)
    while any(len(v) < per_class for v in images_by_class.values()) and seen_batches < max_batches:
        x_batch, y_batch = next(test_gen)
        for img, label in zip(x_batch, y_batch):
            true_name = idx_to_class[int(np.argmax(label))]
            if len(images_by_class[true_name]) < per_class:
                images_by_class[true_name].append(img)
        seen_batches += 1
    images, true_idx = [], []
    for class_name, imgs in images_by_class.items():
        images.extend(imgs)
        true_idx.extend([class_indices[class_name]] * len(imgs))
    return np.array(images), true_idx, idx_to_class


def chart_sample_predictions_sigmoid(n=12, model_label="sigmoid"):
    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{model_label}.keras")
    if not os.path.exists(model_path):
        print(f"{model_path} not found — skipping sample predictions chart.")
        return
    model = load_model(model_path)
    preprocess_fn = get_preprocess_fn()
    raw_test_gen = get_raw_test_generator()
    per_class = n // len(raw_test_gen.class_indices)
    raw_images, true_idx, idx_to_class = _pick_balanced_samples(raw_test_gen, per_class)
    model_ready = preprocess_fn(raw_images.copy() * 255.0)
    preds = model.predict(model_ready, verbose=0)
    n = len(raw_images)
    cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i in range(n):
        pred_idx = int(np.argmax(preds[i]))
        confidence = preds[i][pred_idx]
        correct = true_idx[i] == pred_idx
        axes[i].imshow(raw_images[i]); axes[i].axis("off")
        color = "green" if correct else "red"
        axes[i].set_title(f"True: {idx_to_class[true_idx[i]]}\nPred: {idx_to_class[pred_idx]} ({confidence:.2f})",
                           color=color, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    _save(fig, "09_sample_predictions_sigmoid.png")


def chart_sample_predictions_softmax(n=12, model_label="softmax"):
    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{model_label}.keras")
    if not os.path.exists(model_path):
        print(f"{model_path} not found — skipping sample predictions chart.")
        return
    model = load_model(model_path)
    preprocess_fn = get_preprocess_fn()
    raw_test_gen = get_raw_test_generator()
    per_class = n // len(raw_test_gen.class_indices)
    raw_images, true_idx, idx_to_class = _pick_balanced_samples(raw_test_gen, per_class)
    model_ready = preprocess_fn(raw_images.copy() * 255.0)
    preds = model.predict(model_ready, verbose=0)
    n = len(raw_images)
    cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i in range(n):
        pred_idx = int(np.argmax(preds[i]))
        confidence = preds[i][pred_idx]
        correct = true_idx[i] == pred_idx
        axes[i].imshow(raw_images[i]); axes[i].axis("off")
        color = "green" if correct else "red"
        axes[i].set_title(f"True: {idx_to_class[true_idx[i]]}\nPred: {idx_to_class[pred_idx]} ({confidence:.2f})",
                           color=color, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    _save(fig, "10_sample_predictions_softmax.png")


def chart_gradcam_sigmoid(n=8, model_label="sigmoid"):
    from gradcam import make_gradcam_heatmap, overlay_heatmap
    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{model_label}.keras")
    if not os.path.exists(model_path):
        print(f"{model_path} not found — skipping Grad-CAM chart.")
        return
    model = load_model(model_path)
    preprocess_fn = get_preprocess_fn()
    raw_test_gen = get_raw_test_generator()
    per_class = n // len(raw_test_gen.class_indices)
    raw_images, true_idx, idx_to_class = _pick_balanced_samples(raw_test_gen, per_class)
    n = len(raw_images)
    cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i in range(n):
        img_0_1 = raw_images[i]
        model_ready = preprocess_fn(img_0_1[np.newaxis].copy() * 255.0)
        pred = model.predict(model_ready, verbose=0)[0]
        pred_idx = int(np.argmax(pred))
        heatmap = make_gradcam_heatmap(model, model_ready, class_index=pred_idx)
        overlay = overlay_heatmap(img_0_1, heatmap)
        axes[i].imshow(overlay); axes[i].axis("off")
        correct = true_idx[i] == pred_idx
        color = "green" if correct else "red"
        axes[i].set_title(f"True: {idx_to_class[true_idx[i]]} | Pred: {idx_to_class[pred_idx]} ({pred[pred_idx]:.2f})",
                           color=color, fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Grad-CAM — {model_label}", y=1.02)
    fig.tight_layout()
    _save(fig, "11_gradcam_sigmoid.png")


def chart_gradcam_softmax(n=8, model_label="softmax"):
    from gradcam import make_gradcam_heatmap, overlay_heatmap
    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{model_label}.keras")
    if not os.path.exists(model_path):
        print(f"{model_path} not found — skipping Grad-CAM chart.")
        return
    model = load_model(model_path)
    preprocess_fn = get_preprocess_fn()
    raw_test_gen = get_raw_test_generator()
    per_class = n // len(raw_test_gen.class_indices)
    raw_images, true_idx, idx_to_class = _pick_balanced_samples(raw_test_gen, per_class)
    n = len(raw_images)
    cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i in range(n):
        img_0_1 = raw_images[i]
        model_ready = preprocess_fn(img_0_1[np.newaxis].copy() * 255.0)
        pred = model.predict(model_ready, verbose=0)[0]
        pred_idx = int(np.argmax(pred))
        heatmap = make_gradcam_heatmap(model, model_ready, class_index=pred_idx)
        overlay = overlay_heatmap(img_0_1, heatmap)
        axes[i].imshow(overlay); axes[i].axis("off")
        correct = true_idx[i] == pred_idx
        color = "green" if correct else "red"
        axes[i].set_title(f"True: {idx_to_class[true_idx[i]]} | Pred: {idx_to_class[pred_idx]} ({pred[pred_idx]:.2f})",
                           color=color, fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Grad-CAM — {model_label}", y=1.02)
    fig.tight_layout()
    _save(fig, "12_gradcam_softmax.png")


def chart_external_comparison(primary_results, external_results):
    labels = ordered_labels(set(primary_results.keys()) & set(external_results.keys()))
    if not labels:
        return
    x = np.arange(len(labels)); width = 0.35
    primary_accs = [primary_results[l]["test_accuracy"] for l in labels]
    external_accs = [external_results[l]["test_accuracy"] for l in labels]
    fig, ax = plt.subplots(figsize=(6 + len(labels), 5))
    ax.bar(x - width / 2, primary_accs, width, label="Primary (Kermany) test set")
    ax.bar(x + width / 2, external_accs, width, label="External (NIH ChestX-ray14) test set")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title("Generalization: Primary vs. External Test Set"); ax.legend()
    for i, (p, e) in enumerate(zip(primary_accs, external_accs)):
        ax.text(i - width / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, e + 0.01, f"{e:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, "13_external_comparison.png")


def main():
    eval_path = os.path.join(config.METRICS_DIR, "test_evaluation.json")
    robustness_path = os.path.join(config.METRICS_DIR, "robustness_evaluation.json")
    external_path = os.path.join(config.METRICS_DIR, "external_evaluation.json")

    train_gen, val_gen, test_gen = get_generators()

    chart_training_curves()
    chart_dataset_composition(train_gen, val_gen, test_gen)
    chart_filter_progression()

    eval_results = None
    if os.path.exists(eval_path):
        eval_results = load_json(eval_path)
        chart_confusion_matrices(eval_results)
        chart_roc_curves(eval_results)
        chart_pr_curves(eval_results)
        chart_model_comparison(eval_results)
        chart_per_class_recall(eval_results)
    else:
        print("Skipping evaluation-based charts — run evaluate.py first.")

    if os.path.exists(robustness_path):
        chart_robustness(load_json(robustness_path))
    else:
        print("Skipping robustness chart — run robustness_test.py first.")

    chart_sample_predictions_sigmoid()
    chart_sample_predictions_softmax()
    chart_gradcam_sigmoid()
    chart_gradcam_softmax()

    if os.path.exists(external_path):
        external_results = load_json(external_path)
        chart_confusion_matrices(external_results, filename="02b_external_confusion_matrices.png", title_suffix=" (External/NIH)")
        chart_roc_curves(external_results, filename="03b_external_roc_curves.png", title_suffix=" (External/NIH)")
        chart_per_class_recall(external_results, filename="08b_external_per_class_recall.png", title_suffix=" (External/NIH)")
        if eval_results:
            chart_external_comparison(eval_results, external_results)
    else:
        print("Skipping external-dataset charts — run external_dataset_prepare.py then evaluate_external.py first.")

    print("\nAll charts generated in", config.CHARTS_DIR)


if __name__ == "__main__":
    main()
