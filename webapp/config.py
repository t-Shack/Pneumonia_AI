"""
Configuration for the Flask inference app. Self-contained: expects both
trained models and deployment_config.json to be copied into ./models/
(see README.md for the copy command).
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("PNEUMONIA_MODELS_DIR", os.path.join(BASE_DIR, "models"))

DEPLOYMENT_CONFIG_PATH = os.path.join(MODELS_DIR, "deployment_config.json")
SIGMOID_MODEL_PATH = os.path.join(MODELS_DIR, "pneumonia_sigmoid.keras")
SOFTMAX_MODEL_PATH = os.path.join(MODELS_DIR, "pneumonia_softmax.keras")

CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Chart files expected in static/charts/ (see model_training/README.md for
# how these get generated). Listed here, in display order, so the dashboard
# template doesn't need to guess what exists.
DASHBOARD_CHARTS = [
    ("01_training_curves.png", "Training / validation curves"),
    ("02_confusion_matrices.png", "Confusion matrices"),
    ("03_roc_curves.png", "ROC curves"),
    ("04_precision_recall_curves.png", "Precision-recall curves"),
    ("05_model_comparison.png", "Sigmoid vs. softmax — test accuracy & loss"),
    ("06_robustness_degradation.png", "Robustness under image degradation"),
    ("07_dataset_composition.png", "Dataset class distribution"),
    ("08_filter_progression.png", "CNN filter progression"),
    ("09_sample_predictions.png", "Sample predictions"),
]
