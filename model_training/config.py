"""
Central configuration for the pneumonia detection pipeline — v2, transfer
learning on an ImageNet-pretrained backbone (DenseNet121 by default).
"""

import os

# ---------------------------------------------------------------------------
# Paths — primary training dataset (Kermany/Mooney chest_xray, from Kaggle)
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("PNEUMONIA_DATA_DIR", "./data/chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ---------------------------------------------------------------------------
# Paths — external test set (NIH ChestX-ray14), built by
# external_dataset_prepare.py. Used ONLY for evaluation, never for training.
# ---------------------------------------------------------------------------
NIH_CSV_PATH = os.environ.get("NIH_CSV_PATH", "./data/nih_raw/Data_Entry_2017.csv")
# NIH_RAW_IMAGES_DIR = os.environ.get("NIH_RAW_IMAGES_DIR", "./data/nih_raw/images")
EXTERNAL_TEST_DIR = os.environ.get("EXTERNAL_TEST_DIR", "./data/external_nih_chestxray14")

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
CV_DIR = os.path.join(OUTPUT_DIR, "cross_validation")
GRADCAM_DIR = os.path.join(CHARTS_DIR, "gradcam")

for d in (MODELS_DIR, HISTORY_DIR, METRICS_DIR, CHARTS_DIR, CV_DIR, GRADCAM_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Backbone / transfer learning
# ---------------------------------------------------------------------------
BACKBONE = "densenet121"
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.05

USE_AUGMENTATION = True

WARMUP_EPOCHS = 5
WARMUP_LR = 1e-3

FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_FRACTION = 0.25

EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MONITOR = "val_loss"

# ---------------------------------------------------------------------------
# Loss function / class-imbalance handling
# ---------------------------------------------------------------------------
LOSS_FUNCTION = "focal"
FOCAL_GAMMA = 2.0
# "class_balanced" (recommended, fixes the NORMAL-recall problem — alpha is
#   computed per-class from the actual training distribution, higher for
#   the rarer class) or "fixed" (old behavior: one flat alpha for every
#   class, kept only for an ablation comparison if you want one).
FOCAL_ALPHA_STRATEGY = "class_balanced"
FOCAL_ALPHA_FIXED = 0.25  # only used if FOCAL_ALPHA_STRATEGY == "fixed"
USE_CLASS_WEIGHTS = False

OPTIMIZER = "adam"
METRICS = ["accuracy"]

# ---------------------------------------------------------------------------
# Model packaging
# ---------------------------------------------------------------------------
MODEL_FORMAT = "keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ---------------------------------------------------------------------------
# Cross-validation (optional — see cross_validate.py)
# ---------------------------------------------------------------------------
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Decision threshold calibration (see select_threshold.py)
# ---------------------------------------------------------------------------
# Chosen via Youden's J statistic (max of tpr - fpr) on the VALIDATION set —
# never the test set, since choosing a threshold from test data and then
# reporting test accuracy at that threshold would bias the very number
# being reported.
THRESHOLD_SELECTION_METHOD = "youdens_j"

# ---------------------------------------------------------------------------
# Automatic best-model selection (see select_best_model.py)
# ---------------------------------------------------------------------------
# "accuracy"  -> primary test accuracy alone, ROC-AUC as tiebreaker
# "composite" -> weighted blend of primary accuracy, external accuracy, and
#                mean robustness accuracy (use once you have external +
#                robustness results for every candidate model)
BEST_MODEL_STRATEGY = "accuracy"
COMPOSITE_WEIGHTS = {"primary": 0.5, "external": 0.3, "robustness": 0.2}
