# Pneumonia Detection — Flask App (Phase 2)

Self-contained Flask app: upload a chest X-ray, get predictions from both
the sigmoid and softmax models side by side, and browse the evaluation
charts on `/dashboard`.

## 1. Copy the trained models over

This app doesn't reach into the `model_training/` project at runtime — it
reads from its own `models/` folder. From the `pneumonia-ai-system/` root:

```bash
cp model_training/outputs/models/pneumonia_sigmoid.keras webapp/models/
cp model_training/outputs/models/pneumonia_softmax.keras webapp/models/
cp model_training/outputs/deployment_config.json webapp/models/

cp model_training/outputs/charts/*.png webapp/static/charts/
```

(Windows: use `copy` instead of `cp`, same paths.)

## 2. Install & run

```bash
cd webapp
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

You'll see both models load once at startup (takes a few seconds), then:

```
* Running on http://0.0.0.0:5000
```

Open `http://localhost:5000` to try it locally.

## 3. Get a public URL with ngrok

In a **second terminal**, with the Flask app still running in the first:

```bash
# one-time setup
# 1. Sign up free at https://ngrok.com and grab your authtoken
# 2. Install ngrok (see https://ngrok.com/download for your OS)
ngrok config add-authtoken YOUR_TOKEN_HERE

# every time you want a public link
ngrok http 5000
```

ngrok prints a `https://something.ngrok-free.app` URL — that's what you
share or open on your phone. It stays live as long as both the Flask app
and the `ngrok http 5000` process keep running. Closing either one kills
the link; you'll get a *different* URL the next time you run `ngrok http
5000` unless you're on a paid ngrok plan with a reserved domain.

## 4. What happens on upload

`app.py` → `/predict` reads the image, preprocesses it identically to
training (via `deployment_config.json`, not hardcoded), runs both models,
and returns both verdicts + confidence + full class probabilities. Nothing
is written to disk — the uploaded image is only ever held in memory and
echoed back to the browser as a data URL for display.

## 5. Swapping in the retrained models later

Once you retrain with class weighting, just re-run the `cp` commands from
step 1 to overwrite the three files in `webapp/models/`, then restart
`python app.py`. Nothing else changes — same filenames, same
`deployment_config.json` schema.

## 6. Currently serving the pre-class-weighting models

Right now `webapp/models/` (once you copy them over) holds the biased
models — expect the app to over-call PNEUMONIA on this run. That's fine for
testing the plumbing; swap them per step 5 once retraining finishes.
