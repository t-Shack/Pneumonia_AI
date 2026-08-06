"""
Generates the downloadable PDF report from the same result data already
computed by /predict — no re-inference needed.

Uses xhtml2pdf rather than weasyprint deliberately: pure-Python, pip-only,
no system-level libraries (Pango/Cairo etc.) to install — one less thing to
fight with when this eventually gets deployed to Render or wherever.
Trade-off: xhtml2pdf's CSS support is limited (basic tables/inline styles
work fine; flexbox/grid don't) — pdf_report.html is written with that
constraint in mind, deliberately simpler than the live web page.
"""

import io

from flask import render_template
from xhtml2pdf import pisa


def generate_pdf(result: dict, institution: dict) -> bytes:
    html = render_template("pdf_report.html", result=result, institution=institution)
    buf = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html), dest=buf)
    if pisa_status.err:
        raise RuntimeError(f"PDF generation failed ({pisa_status.err} errors).")
    return buf.getvalue()
