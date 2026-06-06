import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Palette ───────────────────────────────────────────────────────────────────
COLOR_DARK       = colors.HexColor("#0a1020")
COLOR_BLUE       = colors.HexColor("#1a3a6a")
COLOR_BLUE_LIGHT = colors.HexColor("#2a6ab5")
COLOR_TEXT       = colors.HexColor("#2d3748")
COLOR_MUTED      = colors.HexColor("#718096")
COLOR_ALERT      = colors.HexColor("#e07040")
COLOR_DIVIDER    = colors.HexColor("#e2e8f0")
COLOR_DIVIDER_SM = colors.HexColor("#e8edf2")
COLOR_ALERT_BG   = colors.HexColor("#fff5f0")


def generate_pdf_report(conversation: list, collection_name: str = None) -> bytes:
    """
    Génère un rapport PDF médical structuré à partir de l'historique MedRAG.

    Args:
        conversation   : liste de messages {"role", "content", "sources", "alert"}
        collection_name: nom du dossier patient (titre du rapport)

    Returns:
        bytes du PDF prêt à télécharger / streamer
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    s_title    = ParagraphStyle("t",   fontSize=20, fontName="Helvetica-Bold",    textColor=COLOR_DARK,       spaceAfter=4)
    s_subtitle = ParagraphStyle("sub", fontSize=10, fontName="Helvetica",         textColor=COLOR_MUTED,      spaceAfter=2)
    s_meta     = ParagraphStyle("m",   fontSize=8,  fontName="Helvetica",         textColor=COLOR_MUTED,      alignment=TA_RIGHT)
    s_section  = ParagraphStyle("sec", fontSize=9,  fontName="Helvetica-Bold",    textColor=COLOR_BLUE_LIGHT, spaceBefore=14, spaceAfter=6)
    s_question = ParagraphStyle("q",   fontSize=11, fontName="Helvetica-Bold",    textColor=COLOR_DARK,       spaceBefore=10, spaceAfter=4)
    s_answer   = ParagraphStyle("a",   fontSize=10, fontName="Helvetica",         textColor=COLOR_TEXT,       spaceAfter=6,   leftIndent=12, leading=15)
    s_source   = ParagraphStyle("sr",  fontSize=8,  fontName="Helvetica-Oblique", textColor=COLOR_MUTED,      leftIndent=12,  spaceAfter=2)
    s_alert    = ParagraphStyle("al",  fontSize=9,  fontName="Helvetica-Bold",    textColor=COLOR_ALERT)
    s_footer   = ParagraphStyle("f",   fontSize=8,  fontName="Helvetica",         textColor=COLOR_MUTED,      alignment=TA_CENTER)
    s_white    = ParagraphStyle("w",   fontSize=10, fontName="Helvetica-Bold",    textColor=colors.white)
    s_white_sm = ParagraphStyle("ws",  fontSize=8,  fontName="Helvetica",         textColor=colors.HexColor("#a0c4e0"), alignment=TA_RIGHT)

    story = []
    now           = datetime.now()
    patient_label = collection_name.replace("_", " ").title() if collection_name else "Dossier médical"
    n_questions   = len([m for m in conversation if m.get("role") == "user"])

    # ── En-tête : logo + date ─────────────────────────────────────────────────
    header = Table(
        [[Paragraph("<b>MedRAG</b>", s_title), Paragraph(now.strftime("%d/%m/%Y  %H:%M"), s_meta)]],
        colWidths=[13*cm, 4*cm]
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "BOTTOM"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    story.append(header)
    story.append(Paragraph("Clinical Document Intelligence · Rapport de session", s_subtitle))
    story.append(Spacer(1, 6))

    # ── Bandeau patient ───────────────────────────────────────────────────────
    bandeau = Table(
        [[
            Paragraph(f"📋  {patient_label}", s_white),
            Paragraph(
                f"Généré le {now.strftime('%d/%m/%Y à %H:%M')} · {n_questions} question(s)",
                s_white_sm
            )
        ]],
        colWidths=[10*cm, 7*cm]
    )
    bandeau.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), COLOR_BLUE),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (0,-1),  10),
        ("RIGHTPADDING",  (-1,0),(-1,-1), 10),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(bandeau)
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_DIVIDER))
    story.append(Paragraph("Échanges cliniques", s_section))

    # ── Corps : Q&A ───────────────────────────────────────────────────────────
    q_index  = 0
    messages = [m for m in conversation if m.get("role") in ("user", "assistant")]
    i = 0

    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "user":
            q_index += 1
            story.append(Paragraph(f"Q{q_index} — {msg['content']}", s_question))
            i += 1

        elif msg["role"] == "assistant":
            answer  = msg.get("content", "")
            sources = msg.get("sources", [])
            alert   = msg.get("alert", None)

            # Réponse (conversion basique markdown → ReportLab)
            clean = answer.replace("\n", "<br/>")
            story.append(Paragraph(clean, s_answer))

            # Bloc alerte
            if alert and alert.get("has_alert"):
                alerts_text = " · ".join(alert.get("alerts", []))
                level       = alert.get("level", "ATTENTION")
                alert_tbl   = Table(
                    [[Paragraph(f"⚠  Alerte {level} : {alerts_text}", s_alert)]],
                    colWidths=[15*cm]
                )
                alert_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), COLOR_ALERT_BG),
                    ("LEFTPADDING",   (0,0), (-1,-1), 10),
                    ("TOPPADDING",    (0,0), (-1,-1), 6),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                    ("BOX",           (0,0), (-1,-1), 1, COLOR_ALERT),
                ]))
                story.append(alert_tbl)
                story.append(Spacer(1, 4))

            # Sources
            if sources:
                src_txt = "  ·  ".join([f"Page {s.get('page','?')}" for s in sources[:4]])
                story.append(Paragraph(f"Sources : {src_txt}", s_source))

            story.append(HRFlowable(
                width="100%", thickness=0.5, color=COLOR_DIVIDER_SM,
                spaceAfter=4, spaceBefore=8
            ))
            i += 1
        else:
            i += 1

    # ── Pied de page ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_DIVIDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "MedRAG · Clinical Document Intelligence · Abd Elmouhib Souri · ESPRIM 2025<br/>"
        "Ce rapport est généré automatiquement. Il ne remplace pas un avis médical professionnel.",
        s_footer
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()