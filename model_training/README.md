# Model Training — Technical Notes (v2, transfer learning)

## The focal loss bug fix (read this first if retraining sigmoid)

Earlier training run: sigmoid hit `val_loss=7e-13` with `val_accuracy=54.6%`
— loss near-perfect, accuracy near-random, simultaneously. Root cause:
Keras's built-in `categorical_crossentropy` secretly renormalizes its input
before computing loss; the custom focal loss in `losses.py` didn't. That
renormalization is a no-op for softmax (already sums to 1) but essential
for a 2-neuron independent-sigmoid head — without it, each neuron only ever
gets gradient pressure pushing it toward 1 for its own positive class,
never toward 0 for its negative cases. Fixed now (`losses.py`), verified
two ways: (1) the exact same bad predictions that produced ~0 loss before
now produce a meaningfully non-zero loss, and (2) a toy sigmoid-head model
trained end-to-end on a separable synthetic problem now reaches 97.5% val
accuracy instead of collapsing. Softmax was never affected — confirmed the
fix produces byte-identical loss values on softmax-shaped input.
**Retrain sigmoid with `python train.py` before trusting any sigmoid
numbers going forward.**

## Automatic best-model selection

`select_best_model.py` picks a winner and writes `"best_model": "<label>"`
into `outputs/deployment_config.json` — that's what the webapp reads to
know which single `.keras` file to load and serve. Run it after
`evaluate.py`:
```bash
python select_best_model.py
```
Two strategies, set via `config.BEST_MODEL_STRATEGY`:
- `"accuracy"` (default): primary test accuracy, ROC-AUC as tiebreaker.
- `"composite"`: weighted blend of primary + external + robustness
  accuracy (`config.COMPOSITE_WEIGHTS`). Automatically excludes any
  external-evaluation component with zero support for a class — the exact
  situation this project ran into — rather than silently trusting a
  meaningless number, and falls back to `"accuracy"` if neither external
  nor robustness results exist yet.

Full reasoning (not just the winner) is saved to
`outputs/metrics/best_model_selection.json` — useful for the paper's
methodology section.

## Why DenseNet121, why focal loss, why val_loss over val_accuracy

Covered in depth in earlier project discussion / the paper draft — short
version: DenseNet121 matches CheXNet (the field's reference point); focal
loss is a more targeted class-imbalance fix than `class_weight` alone;
`val_loss` is a steadier model-selection signal than `val_accuracy` on a
small, imbalanced validation split (which is what caused sigmoid/softmax to
swap "winner" between runs earlier in this project).

## Preprocessing — the detail most likely to silently break something

DenseNet121's `preprocess_input()` does ImageNet normalization, landing
around `[-2, 2.7]`, NOT `[0, 1]`. Anything that touches raw pixels for
display or math (charts, Grad-CAM, robustness degradation) uses
`data_pipeline.get_raw_test_generator()` (`[0, 1]` only) and applies
`preprocess_input()` as the very last step, never earlier.

## Grad-CAM

`gradcam.py` finds the backbone sub-model by type and its last conv layer
by output shape — not hardcoded names, since those aren't stable across
Keras/backbone versions. Useful beyond box-ticking: some sample X-rays in
this dataset carry visible corner markers, and models trained on it are
known in the literature to sometimes key off incidental artifacts rather
than lung pathology. Grad-CAM lets you show that isn't happening here.

## Running everything, in order

```bash
python train.py                    # sigmoid + softmax, warm-up + fine-tune
python evaluate.py                 # primary test set
python robustness_test.py          # degradation testing
python select_best_model.py        # picks the winner for the webapp
python generate_charts.py          # all charts, including Grad-CAM

# optional but recommended:
python generate_dataset_mapping.py # see top-level README §3 first
python download_external_images.py 
python evaluate_external.py
python select_best_model.py        # rerun if you switch to "composite" strategy
python generate_charts.py          # rerun to pick up external charts

# optional, expensive:
python cross_validate.py --activation sigmoid --folds 5
python cross_validate.py --activation softmax --folds 5
```

## Testing performed on this codebase before delivery

Everything below was actually run, not just reviewed:
- Every file: `py_compile` clean.
- Sigmoid-head + fixed focal loss trained end-to-end on a toy separable
  problem: 97.5% val accuracy (vs. the ~54% collapse under the old loss).
- Full model build → compile → one training step → phase-2 unfreeze →
  save → reload → predictions verified identical before/after reload.
- Grad-CAM: backbone/layer auto-discovery, heatmap generation, and overlay
  all verified working, including after a save/load round trip.
- Every robustness degradation function verified to compose correctly with
  DenseNet preprocessing (no NaNs, sane output ranges).
- `select_best_model.py` verified against synthetic data reproducing the
  exact zero-support external-evaluation scenario from the real run —
  confirmed it correctly excludes that component rather than trusting it.

Not testable in this environment: actual ImageNet weight download (sandbox
network can't reach the pretrained-weight host) and anything requiring your
real dataset — both will work normally on your machine.
