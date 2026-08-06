# Pneumonia Detection — Flask App (v3: single best-model, PDF reports, live Grad-CAM)

## 1. Copy the model over

Only ONE model gets served now (whichever `select_best_model.py` picked),
not both sigmoid and softmax side by side. From the `pneumonia-ai-system/` root:

```bash
cp model_training/outputs/deployment_config.json webapp/models/
```
Then copy **only** the `.keras` file matching `deployment_config.json`'s
`"best_model"` field — e.g. if it says `"best_model": "softmax"`:
```bash
cp model_training/outputs/models/pneumonia_softmax.keras webapp/models/
cp model_training/outputs/charts/*.png webapp/static/charts/
```
If `deployment_config.json` has no `"best_model"` field, run
`model_training/select_best_model.py` first — the app won't start without it.

## 2. Install & run

```bash
cd webapp
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`. Public URL: see the earlier `ngrok` setup
notes (unchanged) — `ngrok http 5000` in a second terminal.

## 3. What's new in this version

- **Single model, not a side-by-side comparison.** The dual sigmoid/softmax
  view is gone from the main app — that comparison still lives in the paper
  and the `/dashboard` charts, but the product now ships one model.
- **Real pages, not an AJAX single-page app.** `/predict` is a genuine form
  POST; on success it redirects to `/result/<id>`, a real page you can
  reload, print, or share the URL of (until it expires).
- **Results are held in memory, not a database.** `result_store.py` is a
  simple in-memory dict keyed by a random ID, evicted after
  `config.RESULT_TTL_SECONDS` (default 30 minutes). This is deliberate —
  prediction history (a real persistent database) was explicitly deferred
  to a later round. Single-process only: fine for `python app.py` or a
  single gunicorn worker; multiple workers won't share this dict, which is
  exactly the point where the deferred history feature becomes necessary.
- **Live Grad-CAM.** Every uploaded image gets its own heatmap, generated
  on the spot (`gradcam.py`, duplicated from `model_training/` so the
  webapp stays deployable standalone) — not pulled from a precomputed chart.
- **PDF export.** `/result/<id>/pdf` re-renders the same stored result data
  as a PDF via `xhtml2pdf` (pure-Python, no system-library dependencies —
  deliberately chosen over `weasyprint` for that reason, given deploying to
  Render is still on the table). `pdf_report.html` is a separate, simpler
  template from the live page — xhtml2pdf's CSS support doesn't cover
  flexbox/grid, so it's plain tables and inline styles.
- **Confidence-tiered clinical language**, not a flat "high confidence"
  regardless of the actual number — see `CONFIDENCE_BANDS` in `config.py`
  and `CLINICAL_RECOMMENDATIONS` in `inference.py` if you want to adjust
  the thresholds or wording.
- **"Local AI processing" language, not "offline."** The site can be
  reached over the internet (that's the whole point of ngrok/Render); what
  IS true and worth saying is that inference itself never calls an external
  API. Check `templates/` and `config.py`'s `INSTITUTION` dict if you want
  to adjust any copy.

## 4. Before you deploy for real

Replace the placeholders in `config.py`'s `INSTITUTION` dict —
`university`, `department`, `researcher`, `email`, `phone` — they're
currently literal `[bracketed placeholder]` strings by design, per your
instruction, and they show up in the footer and the PDF report as-is.

## 5. Tested before delivery

Full request flow verified end-to-end with a real (randomly-initialized,
since this sandbox can't download pretrained weights) DenseNet121 model:
home page render, dashboard render, upload → live inference → Grad-CAM →
redirect to a real result page → PDF download (valid PDF, correct
content-type) → error paths (missing file, expired/invalid result ID) all
returning the right status codes. The real pretrained-weight model will
behave identically, just with real numbers instead of a randomly-initialized
network's.
