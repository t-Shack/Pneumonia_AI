"""
Flask app: upload a chest X-ray, get a full results page (verdict,
confidence, probability chart, Grad-CAM, clinical recommendation, PDF
export), and browse the evaluation charts.

Run:
    python app.py
Then, in a separate terminal, for a public URL:
    ngrok http 5000
"""

import os

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

import config
import inference
import result_store
from pdf_generator import generate_pdf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("home.html", institution=config.INSTITUTION)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("home.html", institution=config.INSTITUTION,
                                error="No file was selected."), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return render_template("home.html", institution=config.INSTITUTION,
                                error="Use a PNG or JPEG image."), 400

    try:
        result = inference.predict_with_gradcam(file.stream)
    except FileNotFoundError as e:
        return render_template("home.html", institution=config.INSTITUTION, error=str(e)), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("home.html", institution=config.INSTITUTION,
                                error=f"Couldn't read that as an image. ({type(e).__name__}: {e})"), 400

    result_id = result_store.save_result(result)
    return redirect(url_for("show_result", result_id=result_id))


@app.route("/result/<result_id>")
def show_result(result_id):
    result = result_store.get_result(result_id)
    if result is None:
        return render_template(
            "home.html", institution=config.INSTITUTION,
            error="That result has expired or wasn't found — please analyze the image again.",
        ), 404
    return render_template("result.html", result=result, institution=config.INSTITUTION, result_id=result_id)


@app.route("/result/<result_id>/pdf")
def download_pdf(result_id):
    result = result_store.get_result(result_id)
    if result is None:
        abort(404)
    pdf_bytes = generate_pdf(result, config.INSTITUTION)
    buf_name = f"pneumonia-report-{result_id[:8]}.pdf"
    from flask import Response
    return Response(
        pdf_bytes, mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={buf_name}"},
    )


@app.route("/dashboard")
def dashboard():
    available = [
        (filename, caption) for filename, caption in config.DASHBOARD_CHARTS
        if os.path.exists(os.path.join(config.CHARTS_DIR, filename))
    ]
    missing_count = len(config.DASHBOARD_CHARTS) - len(available)
    return render_template("dashboard.html", charts=available, missing_count=missing_count,
                            institution=config.INSTITUTION)


if __name__ == "__main__":
    inference.load_the_model()  # load once at startup, not on first request
    app.run(host="0.0.0.0", port=5000, debug=False)
