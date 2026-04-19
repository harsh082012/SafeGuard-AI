import streamlit as st
from analyser import analyse, get_model_info

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SafeGuard AI — Content Analyser",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 600;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    .main-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 300;
        margin-top: 4px;
    }

    .result-safe {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 1px solid #16a34a;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
    }

    .result-toxic {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 1px solid #dc2626;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
    }

    .result-label {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 3px;
        margin: 0;
    }

    .result-safe .result-label  { color: #4ade80; }
    .result-toxic .result-label { color: #f87171; }

    .result-desc {
        font-size: 0.88rem;
        margin-top: 8px;
        font-weight: 300;
    }
    .result-safe .result-desc  { color: #86efac; }
    .result-toxic .result-desc { color: #fca5a5; }

    .metric-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 20px;
        text-align: center;
    }

    .pill-value {
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: #e2e8f0;
        margin: 0;
    }

    .pill-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .category-chip {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 3px;
    }

    .info-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    .batch-safe  { color: #4ade80; font-weight: 600; }
    .batch-toxic { color: #f87171; font-weight: 600; }

    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #4338ca) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover { opacity: 0.85 !important; }

    .divider-line {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 20px 0;
    }

    .sidebar-stat {
        font-family: 'DM Mono', monospace;
        font-size: 0.9rem;
        color: #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    threshold = st.slider(
        "Detection Threshold",
        min_value=0.10, max_value=0.90,
        value=0.50, step=0.05,
        help="Scores above this threshold are classified as toxic. Lower = stricter."
    )

    st.caption(f"Current: `{threshold}` — {'Strict' if threshold < 0.4 else 'Balanced' if threshold < 0.65 else 'Lenient'} mode")

    st.markdown("<hr style='border-color:#1e293b'>", unsafe_allow_html=True)
    st.markdown("### 📊 Model Details")

    try:
        info = get_model_info()
        st.markdown(f"**Architecture** &nbsp; `{info['model']}`")
        st.markdown(f"**ROC-AUC** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class='sidebar-stat'>{info['roc_auc']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Accuracy** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class='sidebar-stat'>{info['accuracy']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Device** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `{info['device']}`")
    except Exception as e:
        st.warning(f"Could not load model info: {e}")

    st.markdown("<hr style='border-color:#1e293b'>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Dataset")
    st.caption("Trained on the **Jigsaw Toxic Comment Classification** dataset — 159,571 Wikipedia comments labelled across 6 toxicity categories.")

    st.markdown("<hr style='border-color:#1e293b'>", unsafe_allow_html=True)
    st.caption("Built with DistilBERT · HuggingFace Transformers · Streamlit")

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">🛡️ SafeGuard AI</p>
    <p class="main-subtitle">Toxic content detection powered by DistilBERT — fine-tuned on the Jigsaw dataset</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e293b; margin: 12px 0 24px 0'>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Analysis", "📋 Batch Analysis"])

# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    text_input = st.text_area(
        "Enter text to analyse:",
        placeholder="Type or paste any comment, message, or post here...",
        height=140,
        label_visibility="visible"
    )

    col_btn, col_clear, col_empty = st.columns([1.4, 1, 3])
    with col_btn:
        run = st.button("🔍 Analyse", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if run:
        if not text_input.strip():
            st.warning("Please enter some text to analyse.")
        else:
            with st.spinner("Analysing content..."):
                result = analyse(text_input, threshold)

            st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

            # Result card
            if result['is_safe']:
                st.markdown(f"""
                <div class="result-safe">
                    <p class="result-label">✓ &nbsp; SAFE</p>
                    <p class="result-desc">This content appears safe and does not violate community guidelines.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-toxic">
                    <p class="result-label">✕ &nbsp; TOXIC</p>
                    <p class="result-desc">This content has been flagged and may violate community guidelines.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-pill">
                    <p class="pill-value">{result['toxic_score']:.4f}</p>
                    <p class="pill-label">Toxic Score</p>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-pill">
                    <p class="pill-value">{result['safe_score']:.4f}</p>
                    <p class="pill-label">Safe Score</p>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-pill">
                    <p class="pill-value">{result['confidence']}%</p>
                    <p class="pill-label">Confidence</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Toxicity bar
            st.markdown("**Toxicity Meter**")
            st.progress(result['meter'] / 100)
            st.caption(f"Score: `{result['toxic_score']:.4f}` vs threshold: `{threshold}`")

            # Categories
            st.markdown("**Assessment**")
            cats_html = "".join([f'<span class="category-chip">{c}</span>' for c in result['categories']])
            st.markdown(cats_html, unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.markdown("Enter one text per line to analyse multiple inputs at once.")

    batch_input = st.text_area(
        "Texts (one per line):",
        placeholder="Hello, how are you?\nYou are an absolute idiot!\nGreat work on the project today.\nI hope you get what you deserve.",
        height=180
    )

    if st.button("🔍 Analyse All", type="primary"):
        lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        else:
            results = []
            prog = st.progress(0, text="Analysing...")
            for i, line in enumerate(lines):
                results.append({**analyse(line, threshold), 'original': line})
                prog.progress((i + 1) / len(lines), text=f"Analysing {i+1}/{len(lines)}...")
            prog.empty()

            safe_n  = sum(1 for r in results if r['is_safe'])
            toxic_n = len(results) - safe_n

            st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

            # Summary row
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Analysed", len(results))
            s2.metric("✅ Safe",  safe_n)
            s3.metric("🚫 Toxic", toxic_n)

            st.markdown("<br>", unsafe_allow_html=True)

            # Results
            import pandas as pd
            rows = []
            for r in results:
                rows.append({
                    "Text":         r['original'][:70] + "..." if len(r['original']) > 70 else r['original'],
                    "Verdict":      r['label'],
                    "Toxic Score":  r['toxic_score'],
                    "Confidence":   f"{r['confidence']}%",
                })
            df = pd.DataFrame(rows)

            def color_verdict(val):
                if val == 'TOXIC': return 'color: #f87171; font-weight: 600'
                if val == 'SAFE':  return 'color: #4ade80; font-weight: 600'
                return ''

            st.dataframe(
                df.style.applymap(color_verdict, subset=['Verdict']),
                use_container_width=True,
                hide_index=True
            )