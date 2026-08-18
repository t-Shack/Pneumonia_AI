"""Central configuration — v3 (audit fixes).
Changes: VALIDATION_SPLIT 0.05->0.20; validation is always clean (enforced in
data_pipeline); threshold default = recall-constrained (not raw Youden);
best-model composite ranks on balanced accuracy; optional mixed-source training."""
import os

DATA_DIR = os.environ.get("PNEUMONIA_DATA_DIR", "./data/rsna_chest_xray") # was ./data/chest_xray (Kermany)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

NIH_CSV_PATH = os.environ.get("NIH_CSV_PATH", "./data/nih_raw/Data_Entry_2017.csv")
EXTERNAL_TEST_DIR = os.environ.get("EXTERNAL_TEST_DIR", "./data/external_nih_chestxray14")
NIH_TRAIN_MIX_DIR = os.environ.get("NIH_TRAIN_MIX_DIR", "./data/nih_train_mix")
USE_MIXED_TRAINING = True  # enable only after populating NIH_TRAIN_MIX_DIR (see README)

OUTPUT_DIR = "./outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
CV_DIR = os.path.join(OUTPUT_DIR, "cross_validation")
GRADCAM_DIR = os.path.join(CHARTS_DIR, "gradcam")
for d in (MODELS_DIR, HISTORY_DIR, METRICS_DIR, CHARTS_DIR, CV_DIR, GRADCAM_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42

BACKBONE = "densenet121"
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
USE_AUGMENTATION = True

WARMUP_EPOCHS = 5
WARMUP_LR = 1e-3
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 1e-5
FINE_TUNE_UNFREEZE_FRACTION = 0.25
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MONITOR = "val_loss"

LOSS_FUNCTION = "focal"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_STRATEGY = "class_balanced"
FOCAL_ALPHA_FIXED = 0.25
USE_CLASS_WEIGHTS = False  # keep False while using focal alpha (else double-counting)
METRICS = ["accuracy"]

MODEL_FORMAT = "keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

CV_FOLDS = 5
CV_DEDUPLICATE_EXACT = True  # drop exact-duplicate / cross-class-conflicting files before folding

THRESHOLD_SELECTION_METHOD = "youdens_j"  # was "youdens_j"
THRESHOLD_MIN_PNEUMONIA_RECALL = 0.95  # floor; among thresholds meeting it, max specificity wins

BEST_MODEL_STRATEGY = "composite"
COMPOSITE_WEIGHTS = {"primary": 0.5, "external": 0.3, "robustness": 0.2}