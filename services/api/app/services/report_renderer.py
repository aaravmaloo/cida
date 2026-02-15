from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def render_pdf(report_payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "CIDA Detection Report")
    y -= 30

    c.setFont("Helvetica", 10)
    for key, value in report_payload.items():
        if isinstance(value, dict):
            c.drawString(40, y, f"{key}:")
            y -= 14
            for k2, v2 in value.items():
                c.drawString(60, y, f"- {k2}: {v2}")
                y -= 14
        else:
            c.drawString(40, y, f"{key}: {value}")
            y -= 14

        if y < 80:
            c.showPage()
            y = height - 50

    c.save()


def render_json(report_payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_payload, ensure_ascii=True, indent=2), encoding="utf-8")

