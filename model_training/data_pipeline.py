"""Data pipeline — v3.
Fixes: validation stream is ALWAYS clean (no augmentation) and identical for
train.py / early stopping / select_threshold.py; explicit stratified split of
the training pool (works with mixed sources); optional Kermany+NIH mixed
training with a filename-level external-test leak guard.
Preprocessing contract unchanged (backbone preprocess baked in; raw [0,1]
generator for robustness/Grad-CAM)."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from model_architecture import get_preprocess_fn

IMG_EXTS = (".png", ".jpg", ".jpeg")


def _dir_dataframe(root):
    rows = []
    for cls in config.CLASS_NAMES:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith(IMG_EXTS):
                rows.append({"filepath": os.path.join(d, fname), "filename": fname, "class": cls})
    return pd.DataFrame(rows)


def get_train_dataframe():
    df = _dir_dataframe(config.TRAIN_DIR)
    if config.USE_MIXED_TRAINING:
        if os.path.isdir(config.NIH_TRAIN_MIX_DIR):
            df = pd.concat([df, _dir_dataframe(config.NIH_TRAIN_MIX_DIR)], ignore_index=True)
        else:
            print(f"USE_MIXED_TRAINING=True but {config.NIH_TRAIN_MIX_DIR} missing - Kermany only.")
        leaked = set()
        if os.path.isdir(config.EXTERNAL_TEST_DIR):
            for cls in config.CLASS_NAMES:
                d = os.path.join(config.EXTERNAL_TEST_DIR, cls)
                if os.path.isdir(d):
                    leaked |= set(os.listdir(d))
        if leaked:
            n = len(df)
            df = df[~df["filename"].isin(leaked)].reset_index(drop=True)
            if len(df) != n:
                print(f"LEAK GUARD: removed {n - len(df)} training file(s) also present in external test set.")
    return df


def get_train_val_split():
    df = get_train_dataframe()
    train_df, val_df = train_test_split(
        df, test_size=config.VALIDATION_SPLIT, stratify=df["class"],
        random_state=config.RANDOM_SEED)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _flow(gen, df, shuffle):
    return gen.flow_from_dataframe(
        df, x_col="filepath", y_col="class", target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE, class_mode="categorical",
        classes=config.CLASS_NAMES, shuffle=shuffle,
        seed=config.RANDOM_SEED if shuffle else None)


def get_generators(augmented=None):
    augmented = config.USE_AUGMENTATION if augmented is None else augmented
    pf = get_preprocess_fn()
    aug = dict(horizontal_flip=True, rotation_range=10, zoom_range=0.1,
               brightness_range=(0.85, 1.15)) if augmented else {}
    train_df, val_df = get_train_val_split()
    train_gen = _flow(ImageDataGenerator(preprocessing_function=pf, **aug), train_df, True)
    val_gen = _flow(ImageDataGenerator(preprocessing_function=pf), val_df, False)  # clean, always
    test_gen = ImageDataGenerator(preprocessing_function=pf).flow_from_directory(
        config.TEST_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES, shuffle=False)
    return train_gen, val_gen, test_gen


def get_validation_eval_generator():
    _, val_df = get_train_val_split()
    return _flow(ImageDataGenerator(preprocessing_function=get_preprocess_fn()), val_df, False)


def get_raw_test_generator():
    return ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        config.TEST_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES, shuffle=False)


def get_external_test_generator():
    return ImageDataGenerator(preprocessing_function=get_preprocess_fn()).flow_from_directory(
        config.EXTERNAL_TEST_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES, shuffle=False)


def print_dataset_summary(train_gen, val_gen, test_gen):
    print("Dataset summary")
    print("-" * 40)
    print(f"Classes: {train_gen.class_indices}")
    print(f"Train images:      {train_gen.samples}")
    print(f"Validation images: {val_gen.samples}")
    print(f"Test images:       {test_gen.samples}")