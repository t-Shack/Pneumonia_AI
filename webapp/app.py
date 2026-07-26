"""
Flask app: upload a chest X-ray, get predictions from both the sigmoid and
softmax models side by side, and browse the evaluation charts.

Run:
    python app.py
Then, in a separate terminal, for a public URL:
    ngrok http 5000
"""

import base64
import io
import os

from flask import Flask, jsonify, render_template, request

import config
import inference

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    available = [
        (filename, caption)
        for filename, caption in config.DASHBOARD_CHARTS
        if os.path.exists(os.path.join(config.CHARTS_DIR, filename))
    ]
    missing_count = len(config.DASHBOARD_CHARTS) - len(available)
    return render_template(
        "dashboard.html", charts=available, missing_count=missing_count
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file was sent."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Use a PNG or JPEG image."}), 400

    raw_bytes = file.read()

    try:
        # Preprocess (for the model) from one copy of the bytes...
        results = inference.predict(
            inference.preprocess_image(io.BytesIO(raw_bytes))
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception:
        return jsonify({"error": "Couldn't read that as an image."}), 400

    # ...and base64-encode a second copy for display, so nothing touches disk.
    image_data_url = "data:image/png;base64," + base64.b64encode(raw_bytes).decode()

    return jsonify({"image": image_data_url, "results": results})


if __name__ == "__main__":
    inference.load_models()  # load once at startup, not on first request
    app.run(host="0.0.0.0", port=5000, debug=False)
