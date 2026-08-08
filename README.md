# Pneumonia Detection — Project README

An AI web app for automated pneumonia identification from chest X-rays.
Two components: **`model_training/`** (builds and evaluates the models) and
**`webapp/`** (Flask app that serves them). This file is the map — where
everything lives, where your data goes, and what order to run things in.

**v2 architecture** (current): transfer learning on an ImageNet-pretrained
DenseNet121 backbone, replacing the original from-scratch 5-block CNN. See
"What changed from v1, and why" at the bottom if you're updating the paper.

---

## 1. Directory structure

```
pneumonia-ai-system/
│
├── model_training/                  <- everything for building/evaluating models
│   │
│   ├── config.py                    <- ALL settings live here: paths, backbone
│   │                                    choice, image size, epochs, loss function,
│   │                                    early stopping, class weights, CV folds
│   ├── model_architecture.py        <- DenseNet121 (or swap: EfficientNetB0/ResNet50)
│   │                                    + classification head; freeze/unfreeze logic
│   ├── losses.py                    <- focal loss (the class-imbalance fix)
│   ├── data_pipeline.py             <- loads + preprocesses images; also exposes
│   │                                    the "raw" [0,1] loader robustness testing needs
│   ├── train.py                     <- trains sigmoid + softmax, two-phase
│   │                                    (warm-up -> fine-tune)
│   ├── evaluate.py                  <- test-set metrics for every model found
│   ├── robustness_test.py           <- noise/blur/contrast/JPEG degradation testing
│   ├── gradcam.py                   <- Grad-CAM heatmaps (interpretability)
│   ├── generate_charts.py           <- every chart/figure, saved as PNGs
│   ├── external_dataset_prepare.py  <- builds the NIH external test set (see §3)
│   ├── evaluate_external.py         <- runs models against that external set
│   ├── cross_validate.py            <- OPTIONAL, expensive: k-fold CV for a
│   │                                    rigor-grade (mean ± std) result
│   ├── requirements.txt
│   ├── README.md                    <- detailed technical notes for this folder
│   │
│   ├── data/                        <- YOU CREATE THIS — see §2 and §3 below
│   │   ├── chest_xray/              <- primary dataset (Kermany/Mooney, Kaggle)
│   │   │   ├── train/{NORMAL,PNEUMONIA}/
│   │   │   └── test/{NORMAL,PNEUMONIA}/
│   │   ├── nih_raw/                 <- external dataset, BEFORE filtering
│   │   │   ├── Data_Entry_2017.csv
│   │   │   └── images/              <- whatever NIH image files you've downloaded
│   │   └── external_nih_chestxray14/ <- external dataset, AFTER filtering
│   │       ├── NORMAL/               (built automatically by
│   │       └── PNEUMONIA/             external_dataset_prepare.py — don't hand-edit)
│   │
│   └── outputs/                     <- EVERYTHING GENERATED lives here, nothing
│       │                                above this line is ever written to by the
│       │                                scripts except this folder
│       ├── models/                  <- pneumonia_sigmoid.keras, pneumonia_softmax.keras
│       ├── history/                 <- per-epoch training logs (csv + json)
│       ├── metrics/                 <- test_evaluation.json, robustness_evaluation.json,
│       │                                external_evaluation.json
│       ├── charts/                  <- every PNG chart, ready for the paper
│       │   └── gradcam/
│       ├── cross_validation/        <- only if you ran cross_validate.py
│       ├── deployment_config.json   <- tells the webapp how to preprocess images
│       │                                identically to training — auto-generated,
│       │                                never hand-edit
│       └── training_summary.json
│
└── webapp/                          <- Flask app, serves the trained models
    ├── app.py                       <- routes: / (upload), /predict, /dashboard
    ├── config.py                    <- paths (self-contained, doesn't read from
    │                                    model_training/ at runtime)
    ├── inference.py                 <- loads models once, preprocesses uploads
    │                                    IDENTICALLY to training (reads
    │                                    deployment_config.json to know how)
    ├── requirements.txt
    ├── README.md                    <- setup + ngrok instructions
    ├── models/                      <- YOU COPY 3 FILES HERE (see §4)
    ├── static/
    │   ├── css/, js/                <- frontend
    │   └── charts/                  <- YOU COPY CHART PNGs HERE (see §4)
    └── templates/                   <- HTML (scan page, dashboard page)
```

---

## 2. Primary dataset — where it goes

Download "Chest X-Ray Images (Pneumonia)" from Kaggle (Kermany/Mooney) and
point `model_training/data/chest_xray/` at it — or set an environment
variable instead of moving files:

```bash
export PNEUMONIA_DATA_DIR=/wherever/you/downloaded/chest_xray
```

Expected structure inside: `train/{NORMAL,PNEUMONIA}/` and
`test/{NORMAL,PNEUMONIA}/`. (Kaggle's `val/` folder is ignored — the
pipeline carves its own 5% validation split out of `train/` instead, since
Kaggle's official `val/` is only 16 images.)

---

## 3. External dataset (NIH ChestX-ray14) — where it goes and how to build it

Used **only for evaluation**, never for training — this is what produces
the cross-institution generalization result (`11_external_comparison.png`).

**Step 1 — get the label file** (small, a few MB, no registration needed):
Download `Data_Entry_2017.csv` from the NIH ChestX-ray14 release (or its
Kaggle mirror, `nih-chest-xrays/data`) into:
```
model_training/data/nih_raw/Data_Entry_2017.csv
```

**Step 2 — get only the images you actually need.** Don't download the
full ~42GB archive. Filter the CSV first:
```bash
cd model_training
python -c "
import pandas as pd
df = pd.read_csv('data/nih_raw/Data_Entry_2017.csv')
pneumonia = df[df['Finding Labels'].str.contains('Pneumonia', na=False)]
normal = df[df['Finding Labels'] == 'No Finding']
print('Pneumonia images needed:', len(pneumonia))   # ~1,431
print('Sample of filenames to fetch:', pneumonia['Image Index'].head().tolist())
"
```
Then pull just those specific files with the Kaggle API (works per-file, not
just whole-dataset):
```bash
kaggle datasets download -d nih-chest-xrays/data -f images_001/images/00000013_005.png -p data/nih_raw/images/
# repeat per filename, or script it from the filtered CSV above

cd model_training
python download_pneumonia_images.py
```
Put every downloaded image somewhere under `data/nih_raw/images/` — the
prep script searches recursively, so exact subfolder layout doesn't matter.

**Step 3 — build the filtered, ready-to-use external test set:**
```bash
python external_dataset_prepare.py
```
This reads the CSV, finds your downloaded images, and writes a clean
`data/external_nih_chestxray14/{NORMAL,PNEUMONIA}/` — labeling any image
with "Pneumonia" among its findings as positive (matches CheXNet's own
convention), "No Finding" as negative, balanced 1:1 by default.

**Step 4 — evaluate against it:**
```bash
python evaluate_external.py
python generate_charts.py     # picks up external results automatically if present
```

Labeling caveat worth a sentence in your methodology: your primary
(Kermany) labels are physician-confirmed; NIH's are NLP-mined from report
text at an estimated >90% accuracy. State it plainly — it's a real
limitation, not a flaw in your design.

---

## 4. Running everything, in order

```bash
cd model_training
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py                 # sigmoid + softmax, warm-up + fine-tune each
python evaluate.py              # primary test set
python robustness_test.py       # degradation testing
python generate_charts.py       # all charts, including Grad-CAM

# optional but recommended:
python generate_dataset_mapping.py    # see §3 first
python download_external_images.py
python evaluate_external.py
python generate_charts.py             # re-run to pick up external charts too

# optional, expensive, for a rigor-grade result instead of one train/test split:
python cross_validate.py --activation sigmoid --folds 5
python cross_validate.py --activation softmax --folds 5
```

Then wire up the webapp:
```bash
cd webapp
cp ../model_training/outputs/models/*.keras models/
cp ../model_training/outputs/deployment_config.json models/
cp ../model_training/outputs/charts/*.png static/charts/
pip install -r requirements.txt
python app.py
```
Full webapp instructions (including the ngrok public-URL steps) are in
`webapp/README.md`.

---

## 5. Before you hit "go" — read this

**Get a GPU sorted first if you can.** Your last training run (from-scratch
CNN, 150×150) was CPU-only and took ~650s/epoch. DenseNet121 at 224×224 is
a deeper network on larger images — expect noticeably longer per-epoch
times on CPU. Google Colab's free tier GPU is the path of least resistance
if fixing local CUDA isn't quick; the code runs identically either way, no
changes needed.

**The first run will download ~30MB of pretrained weights** (DenseNet121,
ImageNet, no-top) automatically via Keras — one-time, cached in
`~/.keras/models/`, needs a normal internet connection (this doesn't work
from a network-restricted sandbox, which is why I could only test the
architecture mechanics with random-initialized weights on my end — the
actual pretrained download will work fine on your machine).

**Config lives in one place.** Every setting mentioned in this README —
backbone choice, image size, epoch counts, loss function, class weight
toggle, CV folds — is in `model_training/config.py`, not scattered across
files.

---

## 6. What changed from v1, and why

| | v1 (previous) | v2 (current) |
|---|---|---|
| Backbone | Custom 5-block CNN, trained from scratch | DenseNet121, ImageNet-pretrained, fine-tuned |
| Input size | 150×150 | 224×224 (DenseNet's native size) |
| Preprocessing | `rescale = 1/255` | Backbone-specific ImageNet normalization |
| Augmentation | Off (kept ablation clean) | On by default |
| Class imbalance fix | `class_weight` | Focal loss (more targeted for this exact failure mode) |
| Model selection | Early stopping on raw `val_accuracy` | Early stopping on `val_loss` (steadier on a small, imbalanced validation split — see model_training/README.md) |
| Interpretability | None | Grad-CAM |
| Generalization check | None | External test set (NIH ChestX-ray14) |
| Rigor option | None | Optional stratified k-fold CV |
| Sigmoid-vs-softmax comparison | Core contribution | **Kept** — same 2-neuron head design, now on the stronger backbone, as a secondary ablation |

The sigmoid-vs-softmax research question survives the rebuild intact; it's
just sitting on a foundation that would actually hold up to review now.
