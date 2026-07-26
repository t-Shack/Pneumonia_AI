"""
Central configuration for the pneumonia detection pipeline — v2, transfer
learning on an ImageNet-pretrained backbone (DenseNet121 by default).

Edit the paths below to match where your datasets live on disk.
"""

import os

# ---------------------------------------------------------------------------
# Paths — primary training dataset (Kermany/Mooney chest_xray, from Kaggle)
# ---------------------------------------------------------------------------
# Expected structure:
#   DATA_DIR/
#       train/{NORMAL,PNEUMONIA}/
#       test/{NORMAL,PNEUMONIA}/
DATA_DIR = os.environ.get("PNEUMONIA_DATA_DIR", "./data/chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ---------------------------------------------------------------------------
# Paths — external test set (NIH ChestX-ray14), built by
# external_dataset_prepare.py. Used ONLY for evaluation, never for training.
# ---------------------------------------------------------------------------
# Where the raw NIH download lives before filtering:
NIH_CSV_PATH = os.environ.get("NIH_CSV_PATH", "./data/nih_raw/Data_Entry_2017.csv")
NIH_RAW_IMAGES_DIR = os.environ.get("NIH_RAW_IMAGES_DIR", "./data/nih_raw/images")
# Where external_dataset_prepare.py writes the filtered, ready-to-use set:
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
# Change this one value to swap backbones (see model_architecture.py for the
# lookup table) — nothing else in the pipeline needs to change.
BACKBONE = "densenet121"

IMG_SIZE = (224, 224)         # DenseNet121's native ImageNet input size
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.05

USE_AUGMENTATION = True       # standard aug is now the default — see README

# Two-phase transfer learning schedule (standard recipe: warm up a fresh
# head on a frozen backbone, THEN unfreeze the top of the backbone and
# fine-tune everything together at a much lower learning rate).
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-3

FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_FRACTION = 0.25   # unfreeze the last 25% of backbone layers

EARLY_STOPPING_PATIENCE = 6
# val_loss, not val_accuracy — raw accuracy on a small, imbalanced
# validation split is what caused sigmoid/softmax to swap "winner" between
# runs earlier in this project. val_loss (especially under focal loss) is a
# steadier signal. See README "Why val_loss, not val_accuracy".
EARLY_STOPPING_MONITOR = "val_loss"

# ---------------------------------------------------------------------------
# Loss function / class-imbalance handling
# ---------------------------------------------------------------------------
# "focal" (recommended) or "categorical_crossentropy"
LOSS_FUNCTION = "focal"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
# Leave this off when using focal loss — the two are largely redundant, and
# combining them is not the standard recipe. Flip to True only if you
# deliberately want class_weight ON TOP of focal loss for a specific reason.
USE_CLASS_WEIGHTS = False

OPTIMIZER = "adam"
METRICS = ["accuracy"]

# ---------------------------------------------------------------------------
# Model packaging
# ---------------------------------------------------------------------------
MODEL_FORMAT = "keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# ---------------------------------------------------------------------------
# Cross-validation (optional — see cross_validate.py). Expensive: trains K
# models per configuration. Off by default given CPU training times; run it
# deliberately, ideally on a GPU, when you want a rigor-grade result rather
# than a single train/test split.
# ---------------------------------------------------------------------------
CV_FOLDS = 5
