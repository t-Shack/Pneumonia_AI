# Training on Google Colab (instead of your local CPU/quota-limited machine)

Two ways to get there: the official VS Code ↔ Colab extension, or Colab's
website directly. Both end up in the same place — training runs on a free
NVIDIA T4 GPU instead of your CPU, and none of it touches your local disk
quota. **Nothing in the codebase needs to change either way.**

---

## Option A — VS Code's official "Google Colab" extension

Google shipped an official extension (Nov 2025) that connects a notebook
open in VS Code to a real Colab-hosted GPU runtime.

1. Extensions view (`Ctrl+Shift+X`) → search **"Google Colab"** → Install
   (official one, published by Google).
2. Create a new file `train_on_colab.ipynb` anywhere in your project.
3. Open it, click the kernel picker (top right of the notebook) → choose
   the Colab option → sign in with your Google account.
4. Pick a runtime with **GPU** (T4) when prompted — this is the step that
   actually gets you the GPU; skipping it silently gives you a CPU runtime.

**Caveat, and why this README has a Plan B:** this extension connects a
notebook to a *remote* Colab kernel — it does not reliably guarantee your
local project folder is visible to that remote machine. Don't assume
`train.py` is just sitting there; the safest pattern is to have the very
first notebook cell pull your project onto Colab's own disk explicitly:

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>/model_training
!pip install -r requirements.txt
```
(No GitHub repo yet? See "Getting your project onto Colab" below for a
zip-upload alternative.)

Then run training in a cell:
```python
!python train.py
```

If the kernel picker doesn't show a Colab option, auth loops without
completing, or the runtime silently stays on CPU — stop fighting it and
switch to Option B. It's not you; it's a fairly new extension.

---

## Option B — Plain Colab website (most reliable fallback)

1. Go to **colab.research.google.com** → New notebook.
2. `Runtime` menu → `Change runtime type` → Hardware accelerator → **T4 GPU** → Save.
3. First cell — get your project onto Colab's disk (pick one):

   **From GitHub (recommended):**
   ```python
   !git clone https://github.com/<your-username>/<your-repo>.git
   %cd <your-repo>/model_training
   ```

   **No repo — upload a zip instead:**
   ```python
   from google.colab import files
   uploaded = files.upload()   # pick your pneumonia-ai-system.zip in the dialog
   !unzip -q pneumonia-ai-system.zip
   %cd pneumonia-ai-system/model_training
   ```

4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```

5. Get your dataset onto Colab — don't re-upload it from your local
   machine, just download it fresh with the Kaggle API (Colab's bandwidth
   makes this fast, and it's a good moment to also grab your NIH external
   test set here instead of fighting your local quota for that too):
   ```python
   from google.colab import files
   files.upload()   # select your kaggle.json API token
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   !unzip -q chest-xray-pneumonia.zip -d data/chest_xray
   ```

6. Run training:
   ```python
   import os
   os.environ["PNEUMONIA_DATA_DIR"] = "/content/<your-repo>/model_training/data/chest_xray/chest_xray"
   !python train.py
   ```

---

## Bringing the results back to your machine

You only need Colab for the GPU-heavy part (`train.py`). Once it's done,
pull just the outputs down and run everything else — `evaluate.py`,
`robustness_test.py`, `generate_charts.py`, external eval — locally as
already documented in the main README; none of those need a GPU.

```python
!zip -r outputs.zip outputs/
from google.colab import files
files.download("outputs.zip")
```
Unzip that into your local `model_training/outputs/`, overwriting the
placeholder folder, and continue from `evaluate.py` onward exactly as
before.

---

## Limits worth knowing about

- **Free tier**: one T4 GPU, 16GB VRAM. Session dies after ~90 minutes idle
  (via the VS Code extension) if VS Code isn't actively connected — keep
  the tab/window open while training runs.
- GPU availability and session length on the free tier fluctuate day to
  day and aren't guaranteed by Google — if you get bumped to CPU or
  disconnected mid-run, just reconnect and rerun; `train.py` doesn't
  resume mid-epoch, so a disconnect means restarting that training call
  (annoying but not corrupting — nothing partial gets saved as a final model).
- If you hit this repeatedly, Colab Pro is a paid step up with longer
  sessions and priority GPU access — not necessary to start, worth knowing
  it exists if free-tier limits become a real blocker.
