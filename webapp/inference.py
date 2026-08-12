"""
Loads the single best-performing model (per deployment_config.json's
"best_model" field, written by model_training/select_best_model.py) once at
startup, and exposes predict_with_gradcam() for the /predict route.
"""

import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.models import load_model

import config
import gradcam

PREPROCESS_FUNCTIONS = {
    "densenet121": densenet_preprocess,
    "efficientnetb0": efficientnet_preprocess,
    "resnet50": resnet50_preprocess,
}

_model = None
_model_label = None
_deployment_config = None


def load_deployment_config():
    global _deployment_config
    if _deployment_config is None:
        if not os.path.exists(config.DEPLOYMENT_CONFIG_PATH):
            raise FileNotFoundError(
                f"deployment_config.json not found at {config.DEPLOYMENT_CONFIG_PATH}. "
                "Copy it from model_training/outputs/ — see README.md."
            )
        with open(config.DEPLOYMENT_CONFIG_PATH) as f:
            _deployment_config = json.load(f)
    return _deployment_config


def load_the_model():
    """Loads the single best model, once. Which model that is comes from
    deployment_config.json's "best_model" field (written by
    model_training/select_best_model.py) — never hardcoded here, so
    retraining/reselecting doesn't require touching webapp code."""
    global _model, _model_label
    if _model is not None:
        return _model

    deployment_config = load_deployment_config()
    label = deployment_config.get("best_model")
    if not label:
        raise RuntimeError(
            "deployment_config.json has no \"best_model\" field. Run "
            "model_training/select_best_model.py, then re-copy deployment_config.json here."
        )

    model_path = os.path.join(config.MODELS_DIR, f"pneumonia_{label}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Copy your trained .keras files into webapp/models/ — see README.md."
        )

    print(f"Loading best model ({label}) from {model_path} ...")
    # compile=False: the webapp only ever calls .predict(), never .evaluate()/.fit(),
    # so no need to deserialize the custom focal loss here at all.
    _model = load_model(model_path, compile=False)
    _model_label = label
    print("Model loaded.")
    return _model


def get_model_label():
    load_the_model()
    return _model_label


def decide_class(probs: np.ndarray, class_indices: dict, threshold) -> int:
    """Argmax was shown to badly favor PNEUMONIA (NORMAL recall ~61% on the
    primary test set despite strong ROC-AUC, and in real testing 3/3 actual
    normal X-rays were misclassified) — a calibration problem, not a
    model-quality one. If deployment_config.json has a calibrated
    "decision_threshold" (written by model_training/select_threshold.py),
    use it directly on the PNEUMONIA probability instead of naive argmax.
    Falls back to argmax if no threshold has been calibrated yet."""
    pneumonia_idx = class_indices["PNEUMONIA"]
    normal_idx = class_indices["NORMAL"]
    if threshold is not None:
        return pneumonia_idx if probs[pneumonia_idx] >= threshold else normal_idx
    return int(np.argmax(probs))


def confidence_band(confidence: float) -> str:
    for threshold, band in config.CONFIDENCE_BANDS:
        if confidence >= threshold:
            return band
    return "low"


CLINICAL_RECOMMENDATIONS = {
    ("PNEUMONIA", "high"): "The AI model predicts pneumonia with high confidence. Clinical evaluation by a qualified healthcare professional is recommended to confirm the diagnosis.",
    ("PNEUMONIA", "moderate"): "The AI model predicts pneumonia with moderate confidence. Clinical evaluation by a qualified healthcare professional is recommended to confirm this finding.",
    ("PNEUMONIA", "low"): "The AI model's prediction of pneumonia has low confidence and should be treated as inconclusive. Clinical evaluation by a qualified healthcare professional is strongly recommended.",
    ("NORMAL", "high"): "No radiographic evidence of pneumonia was detected, with high confidence. Clinical assessment should still be performed where necessary.",
    ("NORMAL", "moderate"): "No radiographic evidence of pneumonia was detected, though model confidence in this result is moderate. Clinical assessment is recommended to confirm.",
    ("NORMAL", "low"): "No radiographic evidence of pneumonia was detected, but model confidence in this result is low and it should be treated as inconclusive. Clinical assessment is strongly recommended.",
}


def clinical_recommendation(predicted_class: str, band: str) -> str:
    return CLINICAL_RECOMMENDATIONS[(predicted_class, band)]


def preprocess_for_model(img_0_1: np.ndarray) -> np.ndarray:
    """img_0_1: (H, W, 3) in [0, 1]. Returns a model-ready (1, H, W, 3) batch."""
    deployment_config = load_deployment_config()
    if "preprocessing" in deployment_config:
        backbone = deployment_config["preprocessing"]
        if backbone not in PREPROCESS_FUNCTIONS:
            raise ValueError(f"Unknown preprocessing={backbone!r} in deployment_config.json.")
        ready = PREPROCESS_FUNCTIONS[backbone](img_0_1[np.newaxis].copy() * 255.0)
    else:
        ready = img_0_1[np.newaxis].copy() * deployment_config["rescale"] * 255.0
    return ready


def load_and_resize(file_stream):
    """Returns (original_pil_image, resized_0_1_array). Keeps the original
    around only for reporting its resolution — never written to disk."""
    deployment_config = load_deployment_config()
    img_size = tuple(deployment_config["img_size"])

    original = Image.open(file_stream).convert("RGB")
    original_resolution = original.size  # (width, height)

    resized = original.resize(img_size)
    arr_0_1 = np.array(resized, dtype=np.float32) / 255.0
    return original_resolution, arr_0_1


def predict_with_gradcam(file_stream):
    """Full pipeline for one uploaded image: preprocess, predict, Grad-CAM
    overlay, all the metadata the results page wants. Returns a plain dict
    (JSON/template-friendly) — nothing here touches disk."""
    start = time.time()

    model = load_the_model()
    deployment_config = load_deployment_config()
    class_indices = deployment_config["class_indices"]
    idx_to_class = {v: k for k, v in class_indices.items()}

    original_resolution, img_0_1 = load_and_resize(file_stream)
    model_ready = preprocess_for_model(img_0_1)

    probs = model.predict(model_ready, verbose=0)[0]
    pred_idx = decide_class(probs, class_indices, deployment_config.get("decision_threshold"))
    predicted_class = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    # Normalize the probability pair to sum to 100% for the doughnut chart —
    # exact for softmax (already sums to 1), a light normalization for
    # sigmoid (independent outputs) so the chart is still readable.
    raw_probs = {idx_to_class[i]: float(p) for i, p in enumerate(probs)}
    prob_sum = sum(raw_probs.values()) or 1.0
    normalized_probs = {k: v / prob_sum for k, v in raw_probs.items()}

    heatmap = gradcam.make_gradcam_heatmap(model, model_ready, class_index=pred_idx)
    overlay = gradcam.overlay_heatmap(img_0_1, heatmap)
    gradcam_data_url = _array_to_data_url(overlay)

    original_data_url = _array_to_data_url((img_0_1 * 255).astype(np.uint8))

    elapsed_ms = (time.time() - start) * 1000

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_band": confidence_band(confidence),
        "clinical_recommendation": clinical_recommendation(predicted_class, confidence_band(confidence)),
        "probabilities": normalized_probs,
        "original_image": original_data_url,
        "gradcam_image": gradcam_data_url,
        "processing_time_ms": round(elapsed_ms, 1),
        "model_label": get_model_label(),
        "backbone": deployment_config.get("backbone", "unknown"),
        "input_resolution": f"{deployment_config['img_size'][0]}x{deployment_config['img_size'][1]}",
        "original_resolution": f"{original_resolution[0]}x{original_resolution[1]}",
        "decision_threshold": deployment_config.get("decision_threshold"),
    }


def _array_to_data_url(arr_uint8: np.ndarray) -> str:
    img = Image.fromarray(arr_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
