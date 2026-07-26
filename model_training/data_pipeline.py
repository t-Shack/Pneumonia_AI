"""
Data loading and preprocessing pipeline — v2 (transfer learning).

IMPORTANT preprocessing note: DenseNet121's preprocess_input() does
ImageNet mean/std normalization, NOT a simple /255 rescale — output values
land roughly in [-2, 2.7], not [0, 1]. That matters for robustness_test.py:
degradation functions (noise, blur, contrast, JPEG) are written assuming
[0, 1] images, so they MUST run on raw-rescaled images BEFORE
preprocess_input is applied, never after. This module exposes both:
  - get_generators()             -> model-ready (backbone preprocessing baked in)
  - get_raw_test_generator()     -> [0, 1] only, for robustness_test.py to degrade
  - get_external_test_generator() -> model-ready, points at the NIH external set
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config
from model_architecture import get_preprocess_fn


def get_generators(augmented: bool = None):
    """Returns (train_generator, val_generator, test_generator), all using
    the backbone's own preprocessing (see module docstring) — ready to feed
    straight into model.fit() / model.predict().

    augmented=None uses config.USE_AUGMENTATION (default True now — unlike
    the from-scratch v1 pipeline, this is no longer held back to keep an
    ablation clean, since the sigmoid-vs-softmax comparison doesn't depend
    on augmentation being off).
    """
    augmented = config.USE_AUGMENTATION if augmented is None else augmented
    preprocess_fn = get_preprocess_fn()

    aug_kwargs = {}
    if augmented:
        aug_kwargs = dict(
            horizontal_flip=True,
            rotation_range=10,
            zoom_range=0.1,
            brightness_range=(0.85, 1.15),
        )

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=config.VALIDATION_SPLIT,
        **aug_kwargs,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES,
        subset="training", shuffle=True, seed=config.RANDOM_SEED,
    )
    val_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES,
        subset="validation", shuffle=True, seed=config.RANDOM_SEED,
    )
    # Test set: no shuffle, so predictions line up with generator.classes.
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES, shuffle=False,
    )
    return train_generator, val_generator, test_generator


def get_raw_test_generator():
    """[0, 1]-scaled test set, NOT backbone-preprocessed — for
    robustness_test.py, which degrades images in [0, 1] space before
    applying backbone preprocessing itself as the final step."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        config.TEST_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES, shuffle=False,
    )


def get_external_test_generator():
    """Model-ready generator over the external NIH test set prepared by
    external_dataset_prepare.py. Evaluation only — never used for training."""
    preprocess_fn = get_preprocess_fn()
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    return datagen.flow_from_directory(
        config.EXTERNAL_TEST_DIR, target_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        class_mode="categorical", classes=config.CLASS_NAMES, shuffle=False,
    )


def print_dataset_summary(train_gen, val_gen, test_gen):
    print("Dataset summary")
    print("-" * 40)
    print(f"Classes: {train_gen.class_indices}")
    print(f"Train images:      {train_gen.samples}")
    print(f"Validation images: {val_gen.samples}")
    print(f"Test images:       {test_gen.samples}")
