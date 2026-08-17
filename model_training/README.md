# Model Training — Technical Notes (v2, transfer learning)

## Two fixes for the NORMAL-recall problem (read this first if retraining)

Real-world testing surfaced it plainly: 3 real normal X-rays, all flagged
as pneumonia. The numbers already showed why — NORMAL recall was ~61% on
the primary test set despite a strong ROC-AUC (0.94–0.97), which is the
signature of a model that ranks cases well but has its decision cutoff in
the wrong place, compounded by a real gap in the loss function. Two fixes,
addressing both:

**Fix 1 — class-balanced focal alpha (`losses.py`, needs a retrain).**
The original focal loss paper's alpha is written `alpha_t` specifically
because it's supposed to take a *different* value per class — higher for
the rarer one — to directly counteract class imbalance. This
implementation previously used one flat `alpha=0.25` for every example
regardless of class, which only provided the "focus on hard examples"
benefit (via gamma) and none of the class-balancing benefit alpha exists
for. Fixed: `train.py`'s `compute_focal_alpha()` now computes a per-class
alpha from the actual training distribution (e.g. `[0.743, 0.257]` for
`[NORMAL, PNEUMONIA]`, mirroring the inverse-frequency logic already used
for `class_weight`), and `losses.py` applies it per-class. Verified
numerically: a NORMAL example misclassified now costs ~2.9x more than an
equivalent PNEUMONIA misclassification (matching the alpha ratio exactly),
versus identical cost either way under the old flat alpha. Toggle via
`config.FOCAL_ALPHA_STRATEGY` (`"class_balanced"` default, `"fixed"` for
the old behavior if you want an ablation comparison in the paper).

**Fix 2 — calibrated decision threshold (`select_threshold.py`, no retrain
needed, ships immediately).** Independent of the alpha problem: even a
well-trained model's naive argmax/0.5 cutoff isn't necessarily the right
operating point. Computed directly from your actual results: at Youden's J
(the threshold maximizing `tpr - fpr`), NORMAL recall goes from 61.5% to
94.4% for softmax (83.6%→90.0% would be sigmoid's equivalent), at the cost
of PNEUMONIA recall dropping from 99.2% to 90.0% — a real, deliberate
tradeoff, not a free lunch. Selected on the **validation** set, never the
test set (picking a threshold from test data and then reporting test
accuracy at that threshold would quietly bias the very number being
reported). Run after `select_best_model.py`:
```bash
python select_threshold.py
```
Writes `"decision_threshold"` into `deployment_config.json` (webapp reads
it automatically) and `outputs/metrics/final_evaluation.json` — cite
*that* file's numbers as your final, as-deployed results, not the
argmax-based `test_evaluation.json` (which exists for model comparison,
not final reporting).

Both fixes address the same symptom from different angles and are meant to
stack — retrain with the fixed alpha, then still run threshold calibration
on top of that retrained model, since even a better-trained model benefits
from a properly chosen operating point.

## The earlier focal-loss collapse bug fix (already applied, for reference)

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
python select_threshold.py         # NEW — calibrates the decision cutoff
python generate_charts.py          # all charts, including Grad-CAM

# optional but recommended:
python generate_dataset_mapping.py # see top-level README §3 first
python download_external_images.py 
python evaluate_external.py
python select_best_model.py        # rerun if you switch to "composite" strategy
python select_threshold.py         # rerun if the winning model changed
python generate_charts.py          # rerun to pick up external charts

# optional, expensive:
python cross_validate.py --activation sigmoid --folds 5
python cross_validate.py --activation softmax --folds 5
```

## Testing performed on this codebase before delivery

This round, tested against your actual real trained model files (not just
synthetic/random-init weights):
- Loaded both real `.keras` files directly: confirmed genuine DenseNet121
  backbone (7,039,554 params, 427 layers), correct output shapes, sigmoid
  outputs correctly independent (sum ≠ 1) vs softmax correctly summing to
  exactly 1.0, and Grad-CAM producing a real, varying heatmap on the actual
  trained weights.
- Class-balanced alpha verified numerically: a NORMAL misclassification
  costs exactly the alpha ratio (2.891x) more than an equivalent PNEUMONIA
  one; flat-alpha mode still produces symmetric costs, confirming the
  toggle works both ways.
- `select_threshold.py`'s core logic (`find_threshold`,
  `evaluate_at_threshold`) unit-tested against synthetic data matching your
  real class distribution — confirmed internal consistency (recall computed
  from the ROC curve matches recall recomputed from the resulting confusion
  matrix at that same threshold).

Earlier rounds (still valid, not re-run this pass unless noted above):
every file `py_compile` clean; sigmoid-head + fixed focal loss trained
end-to-end on a toy problem (97.5% val accuracy vs. the ~54% collapse under
the original bug); full model build→compile→train→unfreeze→save→reload
cycle verified identical predictions before/after reload; every robustness
degradation function verified to compose correctly with DenseNet
preprocessing; `select_best_model.py` verified against synthetic data
reproducing the exact zero-support external-evaluation scenario.

Not testable in this environment: actual ImageNet weight download (sandbox
network can't reach the pretrained-weight host) — not needed this round
since you provided the already-trained weights directly.


# Pneumonia detection pipeline — v3

## What changed vs v2 (audit fixes)
1. Validation is ALWAYS clean (no augmentation) and is an explicit stratified
   split (default 20%) shared by train.py, early stopping and threshold calibration.
2. Threshold selection is recall-constrained (pneumonia recall >= 0.95 on val,
   then max specificity) instead of unconstrained Youden's J on ~60 normals.
3. Evaluation reports balanced accuracy + per-class recall; selector ranks on
   balanced accuracy so "predict-everything-pneumonia" models cannot win.
4. CV removes exact duplicates and cross-class label conflicts (Kermany noise)
   and reports NORMAL recall per fold.
5. Robustness 'motion_blur' is a real directional blur.

## Run order
python dedup_audit.py                      # inspect; add --quarantine to act
python train.py
python evaluate.py
python robustness_test.py
python evaluate_external.py
python select_best_model.py
python select_threshold.py                 # re-run AFTER every retrain
python generate_charts.py
python logit_diagnostic.py                 # sanity: per-class probability bands

## Optional mixed-source training (domain shift fix)
1. Build disjoint NIH train-mix mappings (patch in chat): run
   generate_dataset_mapping.py with --out-prefix trainmix_ --exclude-csv
   pneumonia_zip_mapping.csv normal_zip_mapping.csv
2. Add two JOBS rows in download_external_images.py pointing the trainmix CSVs
   at data/nih_train_mix/NORMAL and .../PNEUMONIA, then run it.
3. Set USE_MIXED_TRAINING = True in config.py and retrain from train.py.
The dataloader also enforces filename disjointness with the external test set.


File
Status

config.py
REWRITTEN — 20% val split, recall-constrained threshold default, dedup flag, mixed-training flags

data_pipeline.py
REWRITTEN — clean val always, explicit stratified split, mixed-source + leak guard

select_threshold.py
REWRITTEN — recall-floor threshold selection (kills the 0.387 failure mode)

evaluate.py
REWRITTEN — adds balanced accuracy, per-class recall, as_deployed block

select_best_model.py
REWRITTEN — composite ranks on balanced accuracy, not raw accuracy

cross_validate.py
REWRITTEN — exact-duplicate removal, per-class recall per fold

robustness_test.py
REWRITTEN — real motion blur (directional kernel)

dedup_audit.py
NEW — standalone duplicate/label-conflict audit

README.md
REWRITTEN

generate_charts.py
PATCHED (one new chart function, snippet below)

train.py, evaluate_external.py, losses.py, model_architecture.py, gradcam.py
UNCHANGED — audited correct. The val-augmentation bug is fixed inside data_pipeline, so train.py needs zero edits.

download_external_images.py, generate_dataset_mapping.py, logit_diagnostic.py
UNCHANGED (optional mix-support patch for the mapping script at the end)
