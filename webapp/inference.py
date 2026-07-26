"""
Loads both trained models once at process startup and exposes a single
predict() function used by the /predict route.

Preprocessing here MUST match training exactly (data_pipeline.py in the
model_training project) — that's why it's driven entirely by
deployment_config.json rather than hardcoded a second time. Supports both
schema versions:
  v2 (transfer learning): {"preprocessing": "densenet121", ...} — uses the
      matching keras.applications preprocess_input function.
  v1 (from-scratch CNN):  {"rescale": 0.00392..., ...} — simple /255 scale.
"""

import json
import os

import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.models import load_model

import config

# Mirrors model_training/model_architecture.py's BACKBONES table. Duplicated
# rather than imported because the webapp is meant to be deployed
# standalone, without a dependency on the training project's source tree.
PREPROCESS_FUNCTIONS = {
    "densenet121": densenet_preprocess,
    "efficientnetb0": efficientnet_preprocess,
    "resnet50": resnet50_preprocess,
}

_models = {}
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


def load_models():
    """Loads both models once. Safe to call repeatedly — subsequent calls
    are no-ops. Called at app startup, not per-request, since loading a
    Keras model takes real time."""
    if _models:
        return _models

    deployment_config = load_deployment_config()
    paths = {
        "sigmoid": config.SIGMOID_MODEL_PATH,
        "softmax": config.SOFTMAX_MODEL_PATH,
    }
    for activation, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{activation} model not found at {path}. "
                "Copy your trained .keras files into webapp/models/ — see README.md."
            )
        print(f"Loading {activation} model from {path} ...")
        # Models trained with focal loss need it importable to deserialize —
        # compile=False sidesteps that entirely since the webapp only ever
        # calls .predict(), never .evaluate()/.fit(), so a compiled loss
        # function isn't needed here at all.
        _models[activation] = load_model(path, compile=False)
    print("Both models loaded.")
    return _models


def preprocess_image(file_stream):
    """Resize to img_size, force RGB, then apply whichever preprocessing
    deployment_config.json specifies (backbone-specific ImageNet
    normalization for v2 models, or a plain rescale for older v1 models)."""
    deployment_config = load_deployment_config()
    img_size = tuple(deployment_config["img_size"])

    img = Image.open(file_stream).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img, dtype=np.float32)  # 0-255 scale, as PIL gives it

    if "preprocessing" in deployment_config:
        backbone = deployment_config["preprocessing"]
        if backbone not in PREPROCESS_FUNCTIONS:
            raise ValueError(
                f"deployment_config.json specifies preprocessing={backbone!r}, "
                f"which isn't in PREPROCESS_FUNCTIONS {list(PREPROCESS_FUNCTIONS)}. "
                "Add it to webapp/inference.py's PREPROCESS_FUNCTIONS table."
            )
        arr = PREPROCESS_FUNCTIONS[backbone](arr)
    else:
        # v1 schema fallback
        arr = arr * deployment_config["rescale"]

    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def predict(image_batch):
    """Runs both models on a single preprocessed image batch and returns a
    result dict keyed by activation, using class_indices from
    deployment_config.json (never hardcoded) so label order can't drift out
    of sync with training."""
    deployment_config = load_deployment_config()
    class_indices = deployment_config["class_indices"]          # {"NORMAL": 0, "PNEUMONIA": 1}
    idx_to_class = {v: k for k, v in class_indices.items()}

    models = load_models()
    results = {}
    for activation, model in models.items():
        probs = model.predict(image_batch, verbose=0)[0]        # shape (2,)
        pred_idx = int(np.argmax(probs))
        results[activation] = {
            "predicted_class": idx_to_class[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                idx_to_class[i]: float(p) for i, p in enumerate(probs)
            },
        }
    return results
