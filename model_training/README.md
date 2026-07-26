# Model Training — Technical Notes (v2, transfer learning)

See the top-level `../README.md` first for directory structure, data
locations, and run order. This file covers the *why* behind the config —
useful when you write the methodology section.

## Why DenseNet121

Matches CheXNet (Stanford, 2017), the most-cited chest X-ray classification
paper — using the same backbone makes your numbers directly comparable to
the field's reference point, not just internally consistent. Swappable via
`config.BACKBONE` (`"densenet121"`, `"efficientnetb0"`, or `"resnet50"`) —
`model_architecture.py`'s `BACKBONES` table is the only place that needs a
new entry to add another option.

## Why focal loss instead of (or alongside) class_weight

`class_weight` reweights the loss by class frequency — a blunt instrument.
Focal loss (Lin et al., 2017) down-weights *easy* examples specifically
(ones the model is already confidently right about), regardless of class,
which is a more targeted fix for the exact failure mode this project ran
into: a model that reaches good overall accuracy by defaulting to the
majority class on ambiguous cases. `config.USE_CLASS_WEIGHTS = False` by
default since combining both isn't the standard recipe — flip it on only if
you have a specific reason to stack them.

## Why val_loss, not val_accuracy, for early stopping / checkpoint selection

Directly informed by what actually happened earlier in this project:
retraining the exact same from-scratch sigmoid/softmax setup, unchanged,
three times produced three different "winners" — because `val_accuracy` on
a 260-image, class-imbalanced validation split is a noisy signal, and
`EarlyStopping` was picking whichever epoch happened to score highest on
it. `val_loss` (especially under focal loss, which already accounts for
class imbalance) is steadier. `cross_validate.py` exists for when you want
a number robust to this kind of noise entirely, via k-fold mean ± std
instead of a single split.

## The two-phase training schedule

Standard transfer-learning recipe, not something specific to this project:
1. **Warm-up** (`config.WARMUP_EPOCHS`, `WARMUP_LR`): backbone frozen
   entirely, only the new head (GlobalAveragePooling → Dropout → Dense)
   trains. Fast, and prevents large early gradients from destroying the
   pretrained weights before the head has learned anything sensible.
2. **Fine-tune** (`config.FINE_TUNE_EPOCHS`, `FINE_TUNE_LR`,
   `FINE_TUNE_UNFREEZE_FRACTION`): the top fraction of the backbone
   unfreezes (default: last 25%), and the whole thing trains together at a
   much lower learning rate. BatchNorm layers stay frozen throughout this
   phase even within the unfrozen region — standard practice, since batch
   size 32 is too small to safely re-estimate their running statistics.

`train.py`'s saved history JSON records `phase_boundary_epoch` so
`generate_charts.py` can mark the warm-up → fine-tune transition on the
training curves.

## Preprocessing — the detail most likely to silently break something

DenseNet121's `preprocess_input()` does ImageNet mean/std normalization,
**not** a `/255` rescale — output values land around `[-2, 2.7]`, not
`[0, 1]`. Everything in this codebase that touches raw pixel values is
written around that fact:

- `data_pipeline.get_generators()` returns model-ready data (preprocessing
  baked into the generator via `preprocessing_function`).
- `data_pipeline.get_raw_test_generator()` returns `[0, 1]`-only data — used
  wherever something needs to *look at* the image (charts, Grad-CAM
  overlays) or mathematically operate on it in `[0, 1]` space
  (`robustness_test.py`'s degradation functions: Gaussian noise, blur,
  contrast, JPEG — all of these assume `[0, 1]` input and would silently
  produce nonsense on ImageNet-normalized values).
- The rule throughout: **degrade or display in `[0, 1]`, apply
  `preprocess_input()` as the very last step before the model sees it.**

If you ever add a new script that loads images, follow this same pattern
rather than reaching for a plain rescale.

## Grad-CAM

`gradcam.py` finds the backbone's last spatial (4D-output) layer
automatically (by shape, not a hardcoded name — layer names aren't stable
across Keras/backbone versions) and the nested backbone sub-model by type.
Worth having beyond "current best practice" box-ticking: some sample X-rays
in this dataset carry visible corner markers, and models trained on it are
known in the literature to sometimes key off incidental artifacts rather
than lung pathology. Grad-CAM lets you show — not just assert — that isn't
happening here.

## Config reference (the full list is in `config.py`, commented)

| Setting | Default | Controls |
|---|---|---|
| `BACKBONE` | `"densenet121"` | Which pretrained network |
| `IMG_SIZE` | `(224, 224)` | Must match the backbone's native input |
| `USE_AUGMENTATION` | `True` | Flip/rotate/brightness during training |
| `WARMUP_EPOCHS` / `WARMUP_LR` | `5` / `1e-3` | Phase 1 |
| `FINE_TUNE_EPOCHS` / `FINE_TUNE_LR` | `30` / `1e-5` | Phase 2 |
| `FINE_TUNE_UNFREEZE_FRACTION` | `0.25` | How much of the backbone unfreezes in phase 2 |
| `EARLY_STOPPING_MONITOR` | `"val_loss"` | See rationale above |
| `LOSS_FUNCTION` | `"focal"` | `"focal"` or `"categorical_crossentropy"` |
| `RANDOM_SEED` | `42` | Applied to `random`, `numpy`, and `tf.random` in `train.py` |
| `CV_FOLDS` | `5` | Only used if you run `cross_validate.py` |

## Sigmoid vs. softmax — still a controlled comparison

Both variants share the identical backbone, head structure
(`Dense(2, activation=...)`), data, and hyperparameters — the only
difference is the output activation, exactly as in the original from-scratch
version of this study. That's what makes it a real ablation rather than two
unrelated models.
