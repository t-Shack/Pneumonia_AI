"""
Robustness evaluation under controlled image quality degradation: Gaussian
noise, motion blur, contrast reduction, JPEG compression, three severities
each, for every discovered model.

Degradation happens in [0, 1] space, THEN backbone preprocessing is applied
as the final step — never the other way around, since preprocess_input()
output isn't in [0, 1] and would break every degradation function's math.

Run after train.py:
    python robustness_test.py
"""

import io
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model

import config
import losses  # noqa: F401 — needed for load_model to deserialize focal loss
from data_pipeline import get_raw_test_generator
from evaluate import discover_models
from model_architecture import get_preprocess_fn


def add_gaussian_noise(images, sigma_255):
    sigma = sigma_255 / 255.0
    noisy = images + np.random.normal(0, sigma, images.shape)
    return np.clip(noisy, 0.0, 1.0)


def add_motion_blur(images, sigma_blur):
    blurred = np.empty_like(images)
    for i in range(images.shape[0]):
        for c in range(images.shape[-1]):
            blurred[i, :, :, c] = gaussian_filter(images[i, :, :, c], sigma=sigma_blur)
    return blurred


def reduce_contrast(images, alpha):
    return np.clip(0.5 + (1 - alpha) * (images - 0.5), 0.0, 1.0)


def apply_jpeg_compression(images, quality):
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        img_uint8 = (images[i] * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        recompressed = np.array(Image.open(buf).convert("RGB")) / 255.0
        out[i] = recompressed
    return out


DEGRADATIONS = {
    "gaussian_noise": {"fn": add_gaussian_noise, "levels": {"mild": 5, "moderate": 15, "severe": 25}},
    "motion_blur": {"fn": add_motion_blur, "levels": {"mild": 1.0, "moderate": 2.0, "severe": 3.5}},
    "contrast_reduction": {"fn": reduce_contrast, "levels": {"mild": 0.25, "moderate": 0.5, "severe": 0.75}},
    "jpeg_compression": {"fn": apply_jpeg_compression, "levels": {"mild": 50, "moderate": 25, "severe": 10}},
}


def collect_test_images(test_gen):
    test_gen.reset()
    images, labels = [], []
    for _ in range(len(test_gen)):
        x, y = next(test_gen)
        images.append(x)
        labels.append(y)
    return np.concatenate(images), np.concatenate(labels)


def evaluate_degraded(model, images, labels, degrade_fn, level_value, preprocess_fn):
    degraded = degrade_fn(images, level_value)
    model_ready = preprocess_fn(degraded * 255.0)
    _, acc = model.evaluate(model_ready, labels, verbose=0)
    return acc


def main():
    preprocess_fn = get_preprocess_fn()
    test_gen = get_raw_test_generator()
    images, labels = collect_test_images(test_gen)

    models = discover_models()
    if not models:
        print(f"No pneumonia_*.keras models found in {config.MODELS_DIR}")
        return

    all_results = {}
    for label, model_path in models.items():
        print(f"\n{'='*60}\nRobustness testing {label.upper()}\n{'='*60}")
        model = load_model(model_path)

        baseline_ready = preprocess_fn(images.copy() * 255.0)
        baseline_loss, baseline_acc = model.evaluate(baseline_ready, labels, verbose=0)
        print(f"Baseline (clean) test accuracy: {baseline_acc:.4f}")

        model_results = {"baseline_accuracy": float(baseline_acc), "degradations": {}}

        for degrade_name, spec in DEGRADATIONS.items():
            print(f"\n{degrade_name}:")
            model_results["degradations"][degrade_name] = {}
            for level_name, level_value in spec["levels"].items():
                acc = evaluate_degraded(model, images, labels, spec["fn"], level_value, preprocess_fn)
                drop = baseline_acc - acc
                model_results["degradations"][degrade_name][level_name] = {
                    "param": level_value, "accuracy": float(acc), "accuracy_drop_pp": float(drop * 100),
                }
                print(f"  {level_name:10s} (param={level_value}): acc={acc:.4f}  (drop {drop*100:.1f} pp)")

        all_results[label] = model_results

    out_path = os.path.join(config.METRICS_DIR, "robustness_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved robustness results to {out_path}")


if __name__ == "__main__":
    main()
