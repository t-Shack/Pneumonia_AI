"""Robustness under degradation — v3. Fix: 'motion_blur' is now an actual
directional (horizontal) motion blur kernel, not an isotropic Gaussian blur.
Degradation in [0,1] space, backbone preprocessing applied last (unchanged)."""
import io
import json
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, convolve
from tensorflow.keras.models import load_model
import config
import losses  # noqa: F401
from data_pipeline import get_raw_test_generator
from evaluate import discover_models
from model_architecture import get_preprocess_fn


def add_gaussian_noise(images, sigma_255):
    sigma = sigma_255 / 255.0
    return np.clip(images + np.random.normal(0, sigma, images.shape), 0.0, 1.0)


def add_motion_blur(images, length):
    length = int(length)
    kernel = np.ones((1, length), dtype=np.float64) / length
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        for c in range(images.shape[-1]):
            out[i, :, :, c] = convolve(images[i, :, :, c], kernel, mode="reflect")
    return out


def reduce_contrast(images, alpha):
    return np.clip(0.5 + (1 - alpha) * (images - 0.5), 0.0, 1.0)


def apply_jpeg_compression(images, quality):
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        pil = Image.fromarray((images[i] * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out[i] = np.array(Image.open(buf).convert("RGB")) / 255.0
    return out


DEGRADATIONS = {
    "gaussian_noise": {"fn": add_gaussian_noise, "levels": {"mild": 5, "moderate": 15, "severe": 25}},
    "motion_blur": {"fn": add_motion_blur, "levels": {"mild": 5, "moderate": 11, "severe": 21}},
    "contrast_reduction": {"fn": reduce_contrast, "levels": {"mild": 0.25, "moderate": 0.5, "severe": 0.75}},
    "jpeg_compression": {"fn": apply_jpeg_compression, "levels": {"mild": 50, "moderate": 25, "severe": 10}},
}


def collect_test_images(test_gen):
    test_gen.reset()
    xs, ys = [], []
    for _ in range(len(test_gen)):
        x, y = next(test_gen)
        xs.append(x); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    preprocess_fn = get_preprocess_fn()
    images, labels = collect_test_images(get_raw_test_generator())
    models = discover_models()
    if not models:
        print(f"No pneumonia_*.keras models found in {config.MODELS_DIR}")
        return
    all_results = {}
    for label, path in models.items():
        print(f"\nRobustness: {label.upper()}")
        model = load_model(path)
        base_ready = preprocess_fn(images.copy() * 255.0)
        _, base_acc = model.evaluate(base_ready, labels, verbose=0)
        print(f"Baseline accuracy: {base_acc:.4f}")
        res = {"baseline_accuracy": float(base_acc), "degradations": {}}
        for name, spec in DEGRADATIONS.items():
            res["degradations"][name] = {}
            for lvl, val in spec["levels"].items():
                degraded = spec["fn"](images, val)
                _, acc = model.evaluate(preprocess_fn(degraded * 255.0), labels, verbose=0)
                res["degradations"][name][lvl] = {
                    "param": val, "accuracy": float(acc),
                    "accuracy_drop_pp": float((base_acc - acc) * 100)}
                print(f"  {name}/{lvl}: acc={acc:.4f}")
        all_results[label] = res
    with open(os.path.join(config.METRICS_DIR, "robustness_evaluation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved robustness_evaluation.json")


if __name__ == "__main__":
    main()