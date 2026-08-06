"""
Configuration for the Flask app. Self-contained — reads from ./models/,
not from ../model_training/ at runtime.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("PNEUMONIA_MODELS_DIR", os.path.join(BASE_DIR, "models"))

DEPLOYMENT_CONFIG_PATH = os.path.join(MODELS_DIR, "deployment_config.json")
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# How long a computed result stays available at /result/<id> and for PDF
# download, in seconds. In-memory only (see result_store.py) — a server
# restart clears it, and that's fine: nothing here is meant to be permanent
# storage. Prediction history (a real database) is a deferred future feature.
RESULT_TTL_SECONDS = 30 * 60

# Confidence bands for the clinical-recommendation wording on the results
# page — deliberately not hardcoding "high confidence" regardless of the
# actual number.
CONFIDENCE_BANDS = [
    (0.90, "high"),
    (0.70, "moderate"),
    (0.0, "low"),
]

DASHBOARD_CHARTS = [
    ("01_training_curves.png", "Training / validation curves"),
    ("02_confusion_matrices.png", "Confusion matrices"),
    ("03_roc_curves.png", "ROC curves"),
    ("04_precision_recall_curves.png", "Precision-recall curves"),
    ("05_model_comparison.png", "Sigmoid vs. softmax — test accuracy & loss"),
    ("06_robustness_degradation.png", "Robustness under image degradation"),
    ("07_dataset_composition.png", "Dataset class distribution"),
    ("10_gradcam.png", "Grad-CAM samples"),
    ("11_external_comparison.png", "Primary vs. external test set"),
]

# Institution / footer content — replace these placeholders with the real
# strings before deploying.
INSTITUTION = {
    "university": "[University Name]",
    "department": "[Department]",
    "researcher": "[Researcher's Name]",
    "email": "[contact@email.com]",
    "phone": "[Phone Number]",
}
