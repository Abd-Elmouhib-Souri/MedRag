"""
frontend/ui.py — MedRAG 
Thème : Blanc + Bleu ciel + Rouge + Jaune — Médical propre et professionnel
"""

import streamlit as st
import requests
import tempfile
import os
from datetime import datetime

st.set_page_config(
    page_title="MedRAG — Clinical Intelligence",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset global ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Fond principal blanc ── */
.stApp {
    background-color: #f5f7fa !important;
}

/* ── Sidebar blanc cassé ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 2px solid #e8eef5 !important;
    box-shadow: 2px 0 12px rgba(0,100,200,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0px; }

/* ── Logo bloc ── */
.logo-bloc {
    background: linear-gradient(135deg, #0077cc 0%, #0099ee 100%);
    padding: 20px 16px 18px;
    margin-bottom: 0;
}
.logo-row   { display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.logo-icon  {
    width:38px; height:38px; background:rgba(255,255,255,0.2);
    border-radius:10px; display:flex; align-items:center;
    justify-content:center; font-size:20px;
    border:1px solid rgba(255,255,255,0.4); flex-shrink:0;
}
.logo-title { font-size:17px; font-weight:700; color:#ffffff; letter-spacing:-0.3px; }
.logo-sub   { font-size:11px; color:rgba(255,255,255,0.75); margin-left:48px; margin-top:2px; }
.status-row {
    display:flex; align-items:center; gap:6px;
    margin-top:10px; font-size:11px; color:rgba(255,255,255,0.8);
}
.dot-green {
    width:7px; height:7px; background:#44ee88; border-radius:50%;
    box-shadow:0 0 6px #44ee8899; flex-shrink:0;
}

/* ── Section labels sidebar ── */
.section-label {
    font-size:10px; font-weight:700; color:#aabbd0;
    letter-spacing:1.2px; text-transform:uppercase; padding:14px 16px 6px;
}

/* ── Patient card ── */
.patient-card {
    background:#f0f7ff; border:1px solid #cce4f8;
    border-left:3px solid #0088dd;
    border-radius:8px; padding:10px 12px; margin:4px 12px 8px;
}
.patient-card-name { font-size:13px; font-weight:700; color:#0066bb; margin-bottom:6px; }
.patient-card-row  { display:flex; justify-content:space-between; font-size:10px; margin-bottom:3px; }
.patient-card-label { color:#88aacc; font-weight:500; }
.patient-card-value { color:#336699; text-align:right; max-width:60%; font-weight:600; }
.drug-tag       { display:inline-block; font-size:9px; background:#e8f4ff; border:1px solid #99ccee; color:#2277aa; padding:2px 7px; border-radius:20px; margin:2px 2px 0 0; font-weight:500; }
.drug-tag-alert { display:inline-block; font-size:9px; background:#fff0f0; border:1px solid #ffaaaa; color:#cc3333; padding:2px 7px; border-radius:20px; margin:2px 2px 0 0; font-weight:600; }

/* ── Alertes ── */
.alert-urgence   { background:#fff0f2; border:1px solid #ffb3be; border-left:4px solid #ee1133; }
.alert-critique  { background:#fff3f0; border:1px solid #ffbbaa; border-left:4px solid #ee4422; }
.alert-danger    { background:#fff8f0; border:1px solid #ffcc99; border-left:4px solid #ff8800; }
.alert-attention { background:#fffdf0; border:1px solid #ffee99; border-left:4px solid #ddaa00; }
.alert-bloc { padding:10px 14px; border-radius:0 8px 8px 8px; font-size:12px; margin-top:6px; max-width:85%; margin-bottom:8px; }
.alert-header { font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px; }
.alert-item   { margin-bottom:3px; line-height:1.5; color:#444; }
.drug-alert-banner {
    background:#fff8e8; border:1px solid #ffdd88;
    border-radius:8px; padding:7px 12px; font-size:11px;
    color:#886600; margin-top:5px; max-width:85%;
    font-weight:500;
}

/* ── Bulles conversation ── */
.bubble-user {
    background:linear-gradient(135deg, #0077cc, #0099ee);
    color:#ffffff; padding:11px 16px;
    border-radius:18px 18px 4px 18px;
    font-size:13px; line-height:1.6; max-width:72%;
    margin-left:auto; margin-bottom:8px;
    box-shadow:0 2px 12px rgba(0,120,200,0.25);
}
.bubble-ai {
    background:#ffffff; border:1px solid #ddeeff;
    color:#223344; padding:11px 16px;
    border-radius:18px 18px 18px 4px;
    font-size:13px; line-height:1.6; max-width:82%;
    margin-bottom:6px;
    box-shadow:0 2px 8px rgba(0,80,160,0.08);
}
.bubble-compare {
    background:#f0f8ff; border:1px solid #aad4f5;
    border-left:4px solid #0099ee;
    color:#223344; padding:13px 16px;
    border-radius:0 18px 18px 18px;
    font-size:13px; line-height:1.7; max-width:92%;
    margin-bottom:6px;
    box-shadow:0 2px 10px rgba(0,100,200,0.1);
}

/* ── Sources ── */
.sources-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; margin-bottom:6px; }
.source-tag {
    font-size:10px; background:#e8f4ff; border:1px solid #99ccee;
    color:#2277aa; padding:3px 10px; border-radius:20px;
    font-family:'JetBrains Mono',monospace; font-weight:500;
}

/* ── Topbar ── */
.topbar {
    background:#ffffff; border-bottom:2px solid #e0eef8;
    padding:13px 22px; display:flex; align-items:center;
    justify-content:space-between; margin-bottom:0;
    box-shadow:0 2px 8px rgba(0,80,160,0.07);
}
.topbar-left { display:flex; flex-direction:column; gap:3px; }
.topbar-doc-title { font-size:14px; font-weight:700; color:#0066bb; }
.topbar-doc-meta  { font-size:11px; color:#88aacc; font-weight:500; }
.topbar-badges    { display:flex; gap:8px; align-items:center; }
.alert-badge-top {
    display:inline-flex; align-items:center; gap:5px;
    background:#fff0f2; border:1px solid #ffaaaa;
    color:#cc2244; font-size:11px; padding:5px 12px;
    border-radius:20px; font-weight:700;
}
.compare-badge {
    display:inline-flex; align-items:center; gap:5px;
    background:#e8f4ff; border:1px solid #88ccee;
    color:#0077bb; font-size:11px; padding:5px 12px;
    border-radius:20px; font-weight:600;
}

/* ── Mode buttons ── */
.stButton > button {
    background:#f0f7ff !important;
    border:1.5px solid #99ccee !important;
    color:#0077bb !important;
    font-size:12px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:600 !important;
    border-radius:8px !important;
    transition:all 0.15s !important;
}
.stButton > button:hover {
    background:#0077cc !important;
    color:#ffffff !important;
    border-color:#0077cc !important;
    box-shadow:0 2px 8px rgba(0,120,200,0.25) !important;
}

/* ── Input chat ── */
[data-testid="stChatInput"] {
    background:#ffffff !important;
    border:2px solid #cce4f5 !important;
    border-radius:12px !important;
    box-shadow:0 2px 8px rgba(0,100,200,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    color:#223344 !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:13px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#aabbd0 !important; }

/* ── File uploader ── */
.stFileUploader {
    background:#f5f9ff !important;
    border:2px dashed #99ccee !important;
    border-radius:10px !important;
}
.stFileUploader label { color:#5599cc !important; font-size:12px !important; font-weight:500 !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div {
    background:#ffffff !important;
    border:1.5px solid #cce4f5 !important;
    border-radius:8px !important;
    color:#223344 !important;
}
[data-testid="stSelectbox"] label { color:#5599cc !important; font-size:11px !important; font-weight:600 !important; }

/* ── Multiselect ── */
[data-testid="stMultiSelect"] > div {
    background:#ffffff !important;
    border:1.5px solid #cce4f5 !important;
    border-radius:8px !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background:#f8fcff !important;
    border:1px solid #ddeef8 !important;
    border-radius:8px !important;
}
div[data-testid="stExpander"] summary {
    color:#4488bb !important;
    font-size:11px !important;
    font-weight:600 !important;
}

/* ── Messages success/error ── */
.stSuccess {
    background:#f0fff5 !important;
    border:1px solid #88ddaa !important;
    color:#226644 !important;
    border-radius:8px !important;
    font-size:12px !important;
}
.stError {
    background:#fff0f2 !important;
    border:1px solid #ffaaaa !important;
    color:#cc2244 !important;
    border-radius:8px !important;
    font-size:12px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color:#0077cc !important; }

/* ── Hint bar ── */
.input-hint {
    text-align:center; font-size:10px;
    color:#aabbd0; margin-top:5px; padding-bottom:3px;
    font-weight:500;
}

/* ── Branding strip ── */
.sidebar-footer {
    font-size:10px; color:#aabbd0; font-weight:500;
    padding:12px 16px; border-top:1px solid #eef4fa; margin-top:8px;
}

#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Utilitaires API ──────────────────────────────────────────────────────────
def fetch_collections() -> list:
    try:
        r = requests.get(f"{API_URL}/collections", timeout=5)
        return r.json().get("collections", []) if r.status_code == 200 else []
    except Exception:
        return []

def fetch_summary(collection_name: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}/summaries/{collection_name}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_all_summaries() -> dict:
    try:
        r = requests.get(f"{API_URL}/summaries", timeout=5)
        return r.json().get("summaries", {}) if r.status_code == 200 else {}
    except Exception:
        return {}


# ─── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "messages":              [],
    "alert_count":           0,
    "available_collections": [],
    "active_collection":     None,
    "mode":                  "single",
    "compare_collections":   [],
    "summaries_cache":       {},
    "_prev_single_sel":      None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.available_collections:
    st.session_state.available_collections = fetch_collections()
    if st.session_state.available_collections and not st.session_state.active_collection:
        st.session_state.active_collection  = st.session_state.available_collections[0]
        st.session_state._prev_single_sel   = st.session_state.available_collections[0]

if not st.session_state.summaries_cache:
    st.session_state.summaries_cache = fetch_all_summaries()


# ─── Helper : alertes ─────────────────────────────────────────────────────────
def render_alert(alert: dict):
    if not alert or not alert.get("has_alert"):
        return
    level  = alert.get("level", "ATTENTION")
    alerts = alert.get("alerts", [])
    drugs  = alert.get("high_risk_drugs_found", [])

    level_css = {
        "URGENCE":   "alert-urgence",
        "CRITIQUE":  "alert-critique",
        "DANGER":    "alert-danger",
        "ATTENTION": "alert-attention"
    }.get(level, "alert-attention")

    level_color = {
        "URGENCE":   "#ee1133",
        "CRITIQUE":  "#ee4422",
        "DANGER":    "#ff8800",
        "ATTENTION": "#ddaa00"
    }.get(level, "#ddaa00")

    items_html = "".join([f'<div class="alert-item">• {a}</div>' for a in alerts])
    st.markdown(f"""
    <div class="alert-bloc {level_css}">
        <div class="alert-header" style="color:{level_color}">⚠ ALERTE {level}</div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)

    if drugs:
        drugs_str = "  ·  ".join(drugs[:4])
        st.markdown(f'<div class="drug-alert-banner">💊 Surveillance renforcée requise : <strong>{drugs_str}</strong></div>', unsafe_allow_html=True)


# ─── Helper : fiche patient ───────────────────────────────────────────────────
def render_patient_card(collection_name: str):
    summary = st.session_state.summaries_cache.get(collection_name)
    if not summary:
        summary = fetch_summary(collection_name)
        if summary:
            st.session_state.summaries_cache[collection_name] = summary

    if not summary:
        st.markdown('<div style="padding:4px 16px;font-size:11px;color:#aabbd0;font-weight:500;">Fiche non disponible — réindexez le PDF</div>', unsafe_allow_html=True)
        return

    diag   = summary.get("diagnostic_principal", "—")[:45]
    age    = summary.get("age", "—")
    drugs  = summary.get("medicaments_actifs", [])
    allerg = summary.get("allergies", [])
    nom    = summary.get("nom_patient", collection_name.replace("_"," ").title())

    HIGH_RISK = ["warfar","héparin","morphin","insuline","lithium","metformin","digoxin","amiodar","valproat"]
    drugs_html  = "".join([
        f'<span class="drug-tag-alert">{d[:22]}</span>'
        if any(hr in d.lower() for hr in HIGH_RISK)
        else f'<span class="drug-tag">{d[:22]}</span>'
        for d in drugs[:6]
    ])
    allerg_html = "".join([f'<span class="drug-tag-alert">{a[:22]}</span>' for a in allerg[:3]])

    st.markdown(f"""
    <div class="patient-card">
        <div class="patient-card-name">👤 {nom}</div>
        <div class="patient-card-row">
            <span class="patient-card-label">Âge</span>
            <span class="patient-card-value">{age}</span>
        </div>
        <div class="patient-card-row">
            <span class="patient-card-label">Diagnostic</span>
            <span class="patient-card-value">{diag}</span>
        </div>
        {"<div style='margin-top:6px;font-size:9px;color:#88aacc;text-transform:uppercase;letter-spacing:0.8px;font-weight:700'>Médicaments</div><div style='margin-top:3px'>" + drugs_html + "</div>" if drugs else ""}
        {"<div style='margin-top:6px;font-size:9px;color:#cc4444;text-transform:uppercase;letter-spacing:0.8px;font-weight:700'>⚠ Allergies</div><div style='margin-top:3px'>" + allerg_html + "</div>" if allerg else ""}
    </div>
    """, unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="logo-bloc">
        <div class="logo-row">
            <div class="logo-icon">🏥</div>
            <span class="logo-title">MedRAG</span>
        </div>
        <div class="logo-sub">Clinical Document Intelligence </div>
        <div class="status-row">
            <div class="dot-green"></div>
            Connecté · ChromaDB · llama3 · Alertes ON
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode
    st.markdown('<div class="section-label">Mode d\'analyse</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if st.button("📄 Dossier unique", key="btn_mode_single", use_container_width=True,
                     type="primary" if st.session_state.mode == "single" else "secondary"):
            if st.session_state.mode != "single":
                st.session_state.mode        = "single"
                st.session_state.messages    = []
                st.session_state.alert_count = 0
                st.rerun()
    with col_m2:
        if st.button("🔀 Comparaison", key="btn_mode_compare", use_container_width=True,
                     type="primary" if st.session_state.mode == "compare" else "secondary"):
            if st.session_state.mode != "compare":
                st.session_state.mode        = "compare"
                st.session_state.messages    = []
                st.session_state.alert_count = 0
                st.rerun()

    # Documents
    st.markdown('<div class="section-label">Documents disponibles</div>', unsafe_allow_html=True)
    col_r, col_n = st.columns([3,1])
    with col_r:
        if st.button("🔄 Rafraîchir", key="btn_refresh", use_container_width=True):
            st.session_state.available_collections = fetch_collections()
            st.session_state.summaries_cache       = fetch_all_summaries()
            st.rerun()
    with col_n:
        n = len(st.session_state.available_collections)
        st.markdown(
            f'<div style="font-size:11px;color:#0088cc;padding-top:6px;text-align:center;font-weight:700">'
            f'{n} doc{"s" if n!=1 else ""}</div>',
            unsafe_allow_html=True
        )

    # Mode Single
    if st.session_state.mode == "single":
        if st.session_state.available_collections:
            idx = 0
            if st.session_state.active_collection in st.session_state.available_collections:
                idx = st.session_state.available_collections.index(st.session_state.active_collection)

            def _on_single_change():
                new_val = st.session_state["sel_single"]
                if new_val != st.session_state.active_collection:
                    st.session_state.active_collection = new_val
                    st.session_state._prev_single_sel  = new_val
                    st.session_state.messages          = []
                    st.session_state.alert_count       = 0

            st.selectbox(
                "Document actif",
                options=st.session_state.available_collections,
                index=idx,
                key="sel_single",
                on_change=_on_single_change
            )

            st.markdown('<div class="section-label">Fiche patient</div>', unsafe_allow_html=True)
            render_patient_card(st.session_state.active_collection)
        else:
            st.markdown('<div style="padding:6px 16px;font-size:11px;color:#aabbd0;font-weight:500;">Aucun document indexé.</div>', unsafe_allow_html=True)

    # Mode Compare
    else:
        if st.session_state.available_collections:
            default_sel = [c for c in st.session_state.compare_collections if c in st.session_state.available_collections]

            def _on_compare_change():
                new_val = st.session_state["sel_compare"]
                if new_val != st.session_state.compare_collections:
                    st.session_state.compare_collections = new_val
                    st.session_state.messages            = []
                    st.session_state.alert_count         = 0

            st.multiselect(
                "Sélectionner 2+ dossiers à comparer",
                options=st.session_state.available_collections,
                default=default_sel,
                key="sel_compare",
                on_change=_on_compare_change
            )

            if st.session_state.compare_collections:
                st.markdown('<div class="section-label">Patients sélectionnés</div>', unsafe_allow_html=True)
                for col_name in st.session_state.compare_collections:
                    render_patient_card(col_name)
        else:
            st.markdown('<div style="padding:6px 16px;font-size:11px;color:#aabbd0;">Aucun document indexé.</div>', unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="section-label">Nouveau document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Glissez un PDF médical", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f'<div style="font-size:11px;color:#5599cc;padding:4px 0;font-weight:500">{uploaded_file.name}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("Indexer", key="btn_ingest"):
                with st.spinner("Indexation + fiche patient..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        response = requests.post(
                            f"{API_URL}/ingest",
                            json={"file_path": tmp_path, "collection_name": uploaded_file.name},
                            timeout=180
                        )
                        os.unlink(tmp_path)
                        if response.status_code == 200:
                            data         = response.json()
                            returned_col = data.get("collection_name", uploaded_file.name)
                            st.session_state.available_collections = fetch_collections()
                            st.session_state.active_collection     = returned_col
                            st.session_state._prev_single_sel      = returned_col
                            st.session_state.messages              = []
                            st.session_state.alert_count           = 0
                            if data.get("summary"):
                                st.session_state.summaries_cache[returned_col] = data["summary"]
                            st.success(f"✅ {data.get('chunks_indexed','?')} chunks · Fiche générée")
                            st.rerun()
                        else:
                            st.error(f"❌ {response.text[:120]}")
                    except Exception as e:
                        st.error(f"❌ {str(e)[:80]}")

    # Session
    st.markdown('<div class="section-label" style="margin-top:4px">Session</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("📄 Exporter rapport PDF", key="btn_export", use_container_width=True):
            if not st.session_state.messages:
                st.warning("Aucun échange à exporter.")
            else:
                with st.spinner("Génération du rapport..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/report",
                            json={"conversation": st.session_state.messages,
                                  "collection_name": st.session_state.active_collection},
                            timeout=30
                        )
                        if response.status_code == 200:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="⬇️ Télécharger", data=response.content,
                                file_name=f"rapport_medrag_{ts}.pdf",
                                mime="application/pdf", key="btn_dl"
                            )
                        else:
                            st.error(f"❌ {response.text[:80]}")
                    except Exception as e:
                        st.error(f"❌ {str(e)[:60]}")
    with col2:
        if st.button("🗑", key="btn_clear", help="Effacer la conversation"):
            st.session_state.messages    = []
            st.session_state.alert_count = 0
            st.rerun()

    st.markdown('<div class="sidebar-footer">Abd Elmouhib Souri · ESPRIM · 2025</div>', unsafe_allow_html=True)


# ─── ZONE PRINCIPALE ──────────────────────────────────────────────────────────

# Topbar
if st.session_state.mode == "single":
    active_name = st.session_state.active_collection or "Aucun document sélectionné"
    active_meta = "Dossier médical · Vectorisé · Prêt" if st.session_state.active_collection else "Sélectionnez un document"
    mode_badge  = '<span style="font-size:10px;color:#5599cc;font-weight:600;background:#e8f4ff;padding:4px 10px;border-radius:20px;border:1px solid #99ccee">📄 Dossier unique</span>'
else:
    n_sel       = len(st.session_state.compare_collections)
    active_name = f"Comparaison — {n_sel} dossier(s) sélectionné(s)"
    active_meta = "  ·  ".join(st.session_state.compare_collections[:3]) or "Sélectionnez 2+ dossiers"
    mode_badge  = '<span class="compare-badge">🔀 Mode comparaison</span>'

alert_html = f'<span class="alert-badge-top">⚠ {st.session_state.alert_count} alerte(s)</span>' if st.session_state.alert_count > 0 else ""

st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-doc-title">📋 {active_name}</div>
        <div class="topbar-doc-meta">{active_meta}</div>
    </div>
    <div class="topbar-badges">{mode_badge}{alert_html}</div>
</div>
""", unsafe_allow_html=True)

# Message de bienvenue
if not st.session_state.messages:
    st.markdown("""
    <div style="padding:50px 20px;text-align:center">
        <div style="font-size:52px;margin-bottom:16px">🏥</div>
        <div style="font-size:22px;color:#0077cc;font-weight:800;margin-bottom:8px;letter-spacing:-0.5px">
            MedRAG 
        </div>
        <div style="font-size:13px;color:#0099dd;font-weight:600;margin-bottom:20px">
            Clinical Document Intelligence
        </div>
        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:20px">
            <span style="background:#e8f4ff;border:1px solid #99ccee;color:#2277aa;padding:6px 14px;border-radius:20px;font-size:11px;font-weight:600">📄 Dossier unique</span>
            <span style="background:#e8f4ff;border:1px solid #99ccee;color:#2277aa;padding:6px 14px;border-radius:20px;font-size:11px;font-weight:600">🔀 Comparaison inter-patients</span>
            <span style="background:#fff8e8;border:1px solid #ffdd88;color:#886600;padding:6px 14px;border-radius:20px;font-size:11px;font-weight:600">⚠ Alertes cliniques</span>
            <span style="background:#f0fff5;border:1px solid #88ddaa;color:#226644;padding:6px 14px;border-radius:20px;font-size:11px;font-weight:600">📊 Export rapport PDF</span>
        </div>
        <div style="font-size:12px;color:#aabbd0;font-weight:500">
            Query Rewriting · Recherche hybride ChromaDB + BM25 · FR / EN / AR
        </div>
    </div>
    """, unsafe_allow_html=True)

# Historique des messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="bubble-user">{message["content"]}</div>', unsafe_allow_html=True)
    elif message["role"] == "assistant":
        answer  = message.get("content", "")
        sources = message.get("sources", [])
        alert   = message.get("alert", None)
        mode    = message.get("mode", "single")
        per_pat = message.get("per_patient", {})

        bubble_class = "bubble-compare" if mode == "compare" else "bubble-ai"
        if mode == "compare":
            st.markdown(
                '<div style="font-size:10px;color:#0088cc;margin-bottom:5px;font-weight:700;'
                'text-transform:uppercase;letter-spacing:0.8px">🔀 Analyse comparative</div>',
                unsafe_allow_html=True
            )
        st.markdown(f'<div class="{bubble_class}">{answer}</div>', unsafe_allow_html=True)

        if mode == "compare" and per_pat:
            for pat_name, pat_data in per_pat.items():
                pat_sources = pat_data.get("sources", [])
                if pat_sources:
                    with st.expander(f"Sources — {pat_name.replace('_',' ').title()}"):
                        for s in pat_sources[:3]:
                            st.markdown(
                                f'<div style="font-size:11px;color:#336699;margin-bottom:4px;'
                                f'padding:6px 10px;background:#f5faff;border-radius:6px;'
                                f'border:1px solid #cce4f5">'
                                f'<strong>Page {s.get("page","?")}</strong> · {s.get("content","")[:160]}...</div>',
                                unsafe_allow_html=True
                            )

        render_alert(alert)

        if mode == "single" and sources:
            tags_html = "".join([f'<span class="source-tag">Page {s.get("page","?")} · Contexte</span>' for s in sources[:4]])
            st.markdown(f'<div class="sources-row">{tags_html}</div>', unsafe_allow_html=True)
            with st.expander("Sources documentaires consultées"):
                for s in sources:
                    st.markdown(
                        f'<div style="font-size:11px;color:#336699;margin-bottom:5px;'
                        f'padding:7px 10px;background:#f5faff;border-radius:6px;'
                        f'border:1px solid #cce4f5">'
                        f'<strong>Page {s.get("page","?")}</strong> · {s.get("content","")[:200]}...</div>',
                        unsafe_allow_html=True
                    )


# ─── INPUT ─────────────────────────────────────────────────────────────────────
placeholder = (
    "Ex: Lequel des patients a l'allergie la plus grave ?"
    if st.session_state.mode == "compare"
    else "Ex: Quel est le diagnostic principal du patient ?"
)

if prompt := st.chat_input(placeholder):

    if st.session_state.mode == "single" and not st.session_state.active_collection:
        st.warning("⚠️ Veuillez sélectionner un document.")
        st.stop()
    if st.session_state.mode == "compare" and len(st.session_state.compare_collections) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 dossiers pour la comparaison.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="bubble-user">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner("Analyse en cours..."):
        try:
            if st.session_state.mode == "single":
                res = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt, "collection_name": st.session_state.active_collection},
                    timeout=60
                )
                if res.status_code != 200:
                    st.error(f"❌ Erreur {res.status_code} : {res.text[:200]}")
                    st.stop()
                data    = res.json()
                answer  = data.get("answer",  "Réponse non disponible.")
                sources = data.get("sources", [])
                alert   = data.get("alert",   None)
                if alert and alert.get("has_alert"):
                    st.session_state.alert_count += 1
                st.session_state.messages.append({
                    "role":"assistant","content":answer,
                    "sources":sources,"alert":alert,"mode":"single"
                })
            else:
                res = requests.post(
                    f"{API_URL}/compare",
                    json={"question": prompt, "collection_names": st.session_state.compare_collections},
                    timeout=120
                )
                if res.status_code != 200:
                    st.error(f"❌ Erreur {res.status_code} : {res.text[:200]}")
                    st.stop()
                try:
                    data = res.json()
                except ValueError:
                    st.error("❌ Réponse invalide. Vérifiez les logs FastAPI.")
                    st.stop()
                answer  = data.get("answer",      "Réponse comparative non disponible.")
                per_pat = data.get("per_patient", {})
                alert   = data.get("alert",       None)
                if alert and alert.get("has_alert"):
                    st.session_state.alert_count += 1
                st.session_state.messages.append({
                    "role":"assistant","content":answer,
                    "sources":[],"alert":alert,
                    "mode":"compare","per_patient":per_pat
                })
            st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("❌ Backend non joignable. Vérifiez que FastAPI tourne sur le port 8000.")
        except requests.exceptions.Timeout:
            st.error("❌ Timeout — réessayez.")
        except Exception as e:
            st.error(f"❌ {str(e)[:120]}")

st.markdown(
    '<div class="input-hint">'
    'Alertes cliniques automatiques · Query Rewriting · Recherche hybride ChromaDB + BM25 · FR / EN / AR'
    '</div>',
    unsafe_allow_html=True
)