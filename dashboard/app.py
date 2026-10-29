import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

# Add project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.severity_model import SeverityClassifier
from src.recurrence_model import WardRecurrenceAnalyzer
from src.priority_score import PriorityScorer
from src.baseline_comparison import BaselineComparator

try:
    from src.voice_nlp_processor import VoiceNLPProcessor
    from src.visual_inspector import VisualInspector
    from src.multimodal_complaint import MultimodalComplaintProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

# Page Config
st.set_page_config(
    page_title="Civic Grievance Priority Triage",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Modern Aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stSidebar {
        background-color: #1E293B !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_grievances.csv")
    if not os.path.exists(csv_path):
        st.error(f"Cleaned dataset not found at {csv_path}. Run pipeline first!")
        st.stop()
    df = pd.read_csv(csv_path, low_memory=False)
    if "Grievance Date" in df.columns:
        df["Grievance Date"] = pd.to_datetime(df["Grievance Date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_ward_recurrence():
    rec_path = os.path.join(PROJECT_ROOT, "data", "processed", "ward_recurrence.csv")
    if os.path.exists(rec_path):
        return pd.read_csv(rec_path)
    return None


@st.cache_resource
def load_severity_classifier():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    clf = SeverityClassifier()
    clf.load_model(models_dir)
    return clf


@st.cache_resource
def load_voice_processor():
    if MULTIMODAL_AVAILABLE:
        return VoiceNLPProcessor()
    return None


@st.cache_resource
def load_visual_inspector():
    if MULTIMODAL_AVAILABLE:
        return VisualInspector()
    return None


@st.cache_resource
def load_multimodal_processor():
    if MULTIMODAL_AVAILABLE:
        return MultimodalComplaintProcessor()
    return None


# Ingestion
df = load_data()
ward_rec_df = load_ward_recurrence()
clf = load_severity_classifier()
scorer = PriorityScorer()
comparator = BaselineComparator(scorer)

# Precompute Ward Recurrence Map
ward_map = {}
if ward_rec_df is not None:
    for _, row in ward_rec_df.iterrows():
        ward_map[row["Ward_Name_Clean"]] = {
            "recurrence_score": float(row["recurrence_score"]),
            "recurrence_level": str(row["recurrence_level"])
        }

# Sidebar Navigation
st.sidebar.title("🏛️ Municipal Triage System")
st.sidebar.markdown("**Decision-Support Dashboard**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview",
        "2. Complaint Analysis",
        "3. Priority Queue & FIFO Comparison",
        "4. Ward Recurrence Risk",
        "5. Complaint Predictor",
        "6. Model Performance",
        "7. 🌐 Multimodal Submission"
    ]
)



# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "1. Overview":
    st.title("📊 Municipal Grievance Overview")
    st.markdown("Operational metrics and grievance distribution overview across Bengaluru municipal wards.")
    
    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    total_complaints = len(df)
    pending_count = len(df[df["Normalized_Status"].isin(["Registered", "In Progress"])])
    resolved_count = len(df[df["Normalized_Status"].isin(["Closed", "Resolved"])])
    critical_count = len(df[df["severity_label"] == "CRITICAL"])
    high_count = len(df[df["severity_label"] == "HIGH"])
    avg_aging = df["days_pending"].mean() if "days_pending" in df.columns else 0.0
    
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Grievances</div><div class="metric-value">{total_complaints:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Pending</div><div class="metric-value" style="color:#F59E0B">{pending_count:,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Resolved</div><div class="metric-value" style="color:#10B981">{resolved_count:,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Critical Severity</div><div class="metric-value" style="color:#EF4444">{critical_count:,}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">High Severity</div><div class="metric-value" style="color:#F97316">{high_count:,}</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Days Pending</div><div class="metric-value" style="color:#3B82F6">{avg_aging:.1f}d</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📌 Grievance Distribution by Category")
        cat_counts = df["Category_Clean"].value_counts().head(10).reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.bar(cat_counts, x="Count", y="Category", orientation="h", color="Count", color_continuous_scale="Blues", template="plotly_dark")
        fig_cat.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        st.subheader("🏘️ Top Wards by Grievance Volume")
        ward_counts = df["Ward_Name_Clean"].value_counts().head(10).reset_index()
        ward_counts.columns = ["Ward", "Count"]
        fig_ward = px.bar(ward_counts, x="Ward", y="Count", color="Count", color_continuous_scale="Oranges", template="plotly_dark")
        st.plotly_chart(fig_ward, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("📈 Monthly Grievance Trend (2020-2025)")
        df["YearMonth"] = df["Grievance Date"].dt.to_period("M").astype(str)
        monthly_df = df.groupby("YearMonth").size().reset_index(name="Count")
        fig_trend = px.line(monthly_df, x="YearMonth", y="Count", template="plotly_dark", color_discrete_sequence=["#38BDF8"])
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c_right:
        st.subheader("⚡ Status Breakdown")
        status_df = df["Normalized_Status"].value_counts().reset_index()
        status_df.columns = ["Status", "Count"]
        fig_status = px.pie(status_df, names="Status", values="Count", color_discrete_sequence=["#10B981", "#EF4444", "#F59E0B", "#3B82F6"], template="plotly_dark")
        st.plotly_chart(fig_status, use_container_width=True)

# ==========================================
# PAGE 2: COMPLAINT ANALYSIS
# ==========================================
elif page == "2. Complaint Analysis":
    st.title("🔍 Deep-Dive Complaint Analysis")
    st.markdown("Filter grievances by category, ward, and status to inspect patterns.")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    categories = ["All Categories"] + sorted(list(df["Category_Clean"].dropna().unique()))
    selected_cat = col_f1.selectbox("Select Category", categories)
    
    wards = ["All Wards"] + sorted(list(df["Ward_Name_Clean"].dropna().unique()))
    selected_ward = col_f2.selectbox("Select Ward", wards)
    
    statuses = ["All Statuses"] + sorted(list(df["Normalized_Status"].dropna().unique()))
    selected_status = col_f3.selectbox("Select Status", statuses)
    
    filtered_df = df.copy()
    if selected_cat != "All Categories":
        filtered_df = filtered_df[filtered_df["Category_Clean"] == selected_cat]
    if selected_ward != "All Wards":
        filtered_df = filtered_df[filtered_df["Ward_Name_Clean"] == selected_ward]
    if selected_status != "All Statuses":
        filtered_df = filtered_df[filtered_df["Normalized_Status"] == selected_status]

    st.markdown(f"**Filtered Results:** `{len(filtered_df):,}` complaints found.")
    st.dataframe(
        filtered_df[["Complaint ID", "Category_Clean", "Sub_Category_Clean", "Ward_Name_Clean", "Grievance Date", "severity_label", "Normalized_Status", "days_pending"]].head(500),
        use_container_width=True
    )

# ==========================================
# PAGE 3: PRIORITY QUEUE & FIFO COMPARISON
# ==========================================
elif page == "3. Priority Queue & FIFO Comparison":
    st.title("📋 Daily Dispatch Queue & FIFO Comparison")
    
    tab1, tab2 = st.tabs(["🚀 Ranked Dispatch Worklist", "⚖️ Baseline Comparison (FIFO vs Proposed)"])
    
    with tab1:
        st.markdown("Operational priority worklist ordered by **Composite Priority Score** (Severity 50% + Recurrence 30% + Aging 20%).")
        col1, col2, col3 = st.columns(3)
        p_filter = col1.selectbox("Filter Priority Level", ["All Priority Levels", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        w_filter = col2.selectbox("Filter Ward", ["All Wards"] + sorted(list(df["Ward_Name_Clean"].unique())))
        sample_size = col3.slider("Queue Sample Size", 100, 5000, 1000, step=100)
        
        sample_subset = df.sample(n=min(sample_size, len(df)), random_state=42)
        worklist_df = scorer.generate_dispatch_worklist(sample_subset, ward_map)        
        if p_filter != "All Priority Levels":
            worklist_df = worklist_df[worklist_df["priority_level"] == p_filter]
        if w_filter != "All Wards":
            worklist_df = worklist_df[worklist_df["Ward_Name_Clean"] == w_filter]

        st.subheader(f"Ranked Worklist ({len(worklist_df)} complaints)")
        disp_cols = ["Rank", "Complaint ID", "Category_Clean", "Sub_Category_Clean", "Ward_Name_Clean", "severity_label", "recurrence_score", "days_pending", "priority_score", "priority_level", "explanation"]
        st.dataframe(worklist_df[disp_cols], use_container_width=True)
        
        csv_data = worklist_df[disp_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Ranked Dispatch Queue to CSV",
            data=csv_data,
            file_name="daily_dispatch_worklist.csv",
            mime="text/csv"
        )
        
    with tab2:
        st.subheader("Comparison: Proposed Composite System vs FIFO Baseline")
        st.markdown("Demonstrating how prioritizing by **Composite Priority Score** accelerates resolution of high-risk grievances compared to traditional First-In, First-Out (FIFO).")
        
        comp_sample = df.sample(n=min(500, len(df)), random_state=42)
        comp_res = comparator.compare_methods(comp_sample, ward_map)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("High-Risk in Top 10% (Proposed)", comp_res["high_risk_in_top_proposed"])
        m2.metric("High-Risk in Top 10% (FIFO)", comp_res["high_risk_in_top_fifo"])
        m3.metric("Prioritization Gain Ratio", f"{comp_res['high_risk_gain_ratio']}x")
        
        st.markdown(f"**Average Rank Shift for CRITICAL Complaints:** `+{comp_res['avg_rank_shift_critical']:.1f}` ranks higher in queue under Proposed System.")
        
        st.subheader("Spearman Rank Correlation Analysis")
        st.json(comp_res["spearman_correlations"])
        
        st.subheader("Detailed Rank Comparison Table")
        st.dataframe(comp_res["ranked_dataframe"].head(100), use_container_width=True)

# ==========================================
# PAGE 4: WARD RECURRENCE RISK
# ==========================================
elif page == "4. Ward Recurrence Risk":
    st.title("🗺️ Ward Recurrence Risk Clustering")
    st.markdown("K-Means Ward Recurrence Analysis (Grouping 199 municipal wards into High, Medium, and Low Risk).")
    
    if ward_rec_df is not None:
        c1, c2, c3 = st.columns(3)
        high_risk_wards = len(ward_rec_df[ward_rec_df["recurrence_level"] == "High Recurrence"])
        med_risk_wards = len(ward_rec_df[ward_rec_df["recurrence_level"] == "Medium Recurrence"])
        low_risk_wards = len(ward_rec_df[ward_rec_df["recurrence_level"] == "Low Recurrence"])
        
        c1.metric("High Recurrence Risk Wards", high_risk_wards, delta_color="inverse")
        c2.metric("Medium Recurrence Risk Wards", med_risk_wards)
        c3.metric("Low Recurrence Risk Wards", low_risk_wards, delta_color="normal")
        
        st.markdown("<br>", unsafe_allow_html=True)
        fig_rec = px.bar(
            ward_rec_df.sort_values(by="complaint_count", ascending=False).head(25),
            x="Ward_Name_Clean",
            y="complaint_count",
            color="recurrence_level",
            color_discrete_map={"High Recurrence": "#EF4444", "Medium Recurrence": "#F59E0B", "Low Recurrence": "#10B981"},
            title="Top 25 Wards Recurrence Risk Breakdown",
            template="plotly_dark"
        )
        st.plotly_chart(fig_rec, use_container_width=True)
        
        st.subheader("📍 Interactive Ward Recurrence Map (Bengaluru OpenStreetMap)")
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="CartoDB dark_matter")
        
        ward_coords = {
            "Jnanabharathi Ward": [12.9560, 77.5020],
            "Horamavu": [13.0280, 77.6600],
            "Rajarajeshwari Nagar": [12.9274, 77.5186],
            "Begur": [12.8760, 77.6250],
            "Thanisandra": [13.0550, 77.6250],
            "Hemmigepura": [12.8890, 77.5180],
            "Bellandur": [12.9304, 77.6784],
            "Jakkur": [13.0780, 77.6070],
            "Uttarahalli": [12.9060, 77.5450],
            "Someshwara": [12.9350, 77.6180]
        }
        
        for _, r in ward_rec_df.head(20).iterrows():
            w_name = r["Ward_Name_Clean"]
            if w_name in ward_coords:
                lat, lon = ward_coords[w_name]
                level = r["recurrence_level"]
                color = "red" if level == "High Recurrence" else ("orange" if level == "Medium Recurrence" else "green")
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12,
                    popup=f"<b>{w_name}</b><br>Risk: {level}<br>Complaints: {r['complaint_count']:,}",
                    color=color,
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
                
        st_folium(m, width=1100, height=450)
        st.dataframe(ward_rec_df, use_container_width=True)

# ==========================================
# PAGE 5: COMPLAINT PREDICTOR
# ==========================================
elif page == "5. Complaint Predictor":
    st.title("🎯 Live Grievance Severity & Priority Predictor")
    st.markdown("Test real-time triage scoring for any new incoming municipal complaint.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        cat_input = col1.selectbox("Category", sorted(list(df["Category_Clean"].unique())))
        sub_cat_input = col2.text_input("Sub Category", "Maintenance of Streetlights / Potholes / Drainage")
        
        col3, col4 = st.columns(2)
        ward_input = col3.selectbox("Ward Name", sorted(list(df["Ward_Name_Clean"].unique())))
        date_input = col4.date_input("Complaint Date")
        
        desc_input = st.text_area("Grievance Description / Staff Remarks", "Open manhole on main road causing serious accident danger and traffic collapse")
        submit_btn = st.form_submit_button("🚀 Run Priority Scoring")
        
    if submit_btn:
        full_text = f"{cat_input} {sub_cat_input} {desc_input}".lower()
        pred_sev = clf.predict_severity([full_text])[0]
        
        rec_info = ward_map.get(ward_input, {"recurrence_score": 0.5, "recurrence_level": "Medium Recurrence"})
        rec_score = rec_info["recurrence_score"]
        
        days_p = max(0, (pd.Timestamp.now() - pd.Timestamp(date_input)).days)
        res = scorer.calculate_priority_score(pred_sev, rec_score, days_p)
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Predicted Severity", pred_sev)
        res_col2.metric("Ward Recurrence Risk", rec_info["recurrence_level"], f"{rec_score*100:.0f}%")
        res_col3.metric("Aging Days", f"{days_p} days")
        res_col4.metric("Composite Priority Score", f"{res['priority_score']:.1f} / 100", res["priority_level"])
        
        st.info(f"**Explanation:** {res['explanation']}")

# ==========================================
# PAGE 6: MODEL PERFORMANCE
# ==========================================
elif page == "6. Model Performance":
    st.title("📈 Machine Learning Model Performance")
    st.markdown("Evaluation metrics comparing **Naive Bayes (Primary)** against **Logistic Regression**, **Decision Tree**, and **KNN**.")
    
    metrics_path = os.path.join(PROJECT_ROOT, "models", "severity_metrics_report.joblib")
    if os.path.exists(metrics_path):
        import joblib
        metrics = joblib.load(metrics_path)
        m_df = pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1_score"]]
        
        st.subheader("Classification Performance Comparison")
        st.table(m_df.style.highlight_max(axis=0, color="#1E3A8A"))
        
        fig_m = px.bar(
            m_df.reset_index(),
            x="index",
            y=["accuracy", "precision", "recall", "f1_score"],
            barmode="group",
            title="Algorithm Metrics Comparison",
            template="plotly_dark"
        )
        st.plotly_chart(fig_m, use_container_width=True)

# ==========================================
# PAGE 7: MULTIMODAL SUBMISSION
# ==========================================
elif page == "7. 🌐 Multimodal Submission":
    import tempfile
    import time as _time

    st.title("🌐 Multimodal AI Complaint Submission & Intelligent Triage")
    st.markdown("Submit civic grievances via **regional voice notes**, **photos / videos**, or **combined multimodal inputs** across Indian languages with real-time translation, YOLOv8 visual validation, and municipal department routing.")

    # --- Dependency status banner ---
    _dep_status = {}
    for _pkg, _import_name in [
        ('torch', 'torch'), ('ultralytics', 'ultralytics'),
        ('whisper', 'whisper'), ('transformers', 'transformers'),
        ('langdetect', 'langdetect'), ('cv2', 'cv2'), ('PIL', 'PIL')
    ]:
        try:
            __import__(_import_name)
            _dep_status[_pkg] = True
        except ImportError:
            _dep_status[_pkg] = False

    _missing = [k for k, v in _dep_status.items() if not v]
    with st.expander('🔧 Multimodal AI Engine & Dependency Status', expanded=False):
        _cols = st.columns(len(_dep_status))
        for _i, (_pkg, _ok) in enumerate(_dep_status.items()):
            _cols[_i].markdown(f"{'✅' if _ok else '❌'} **{_pkg}**")
        if _missing:
            st.warning(f"Optional packages missing: `pip install {' '.join(_missing)}`")
        else:
            st.success('All Voice, Vision, and NLP AI dependencies are active and operational.')

    PLACEHOLDER_TEXT = {
        "en": "Type your complaint here in English...",
        "kn": "ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಹೊಂಡ ಇದೆ, ವಾಹನಗಳಿಗೆ ಹಾನಿ ಆಗುತ್ತಿದೆ...",
        "hi": "मुख्य सड़क के पास कचरे का ढेर बहुत बड़ा हो गया है और बदबू आ रही है...",
        "ta": "தெரு விளக்கு கடந்த மூன்று நாட்களாக எரியவில்லை...",
        "te": "పెద్ద చెట్టు రోడ్డుపై పడిపోయింది, వాహన రాకపోకలు నిలిచిపోయాయి...",
    }

    LANGUAGE_OPTIONS = {
        "Auto-detect / Any": "auto",
        "English": "en",
        "Kannada (ಕನ್ನಡ)": "kn",
        "Hindi (हिन्दी)": "hi",
        "Tamil (தமிழ்)": "ta",
        "Telugu (తెలుగు)": "te",
    }

    SAMPLE_VOICE_TEXTS = {
        "-- Type Custom Text / Upload Audio --": "",
        "BBMP-SAMPLE-001 (Kannada): Pothole damaging vehicles": "ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಹೊಂಡ ಇದೆ, ವಾಹನಗಳಿಗೆ ಹಾನಿ ಆಗುತ್ತಿದೆ",
        "BBMP-SAMPLE-002 (Hindi): Overflowing garbage dump": "मुख्य सड़क के पास कचरे का ढेर बहुत बड़ा हो गया है और बदबू आ रही है",
        "BBMP-SAMPLE-003 (Tamil): Streetlight not working": "தெரு விளக்கு கடந்த மூன்று நாட்களாக எரியவில்லை, இரவில் நடக்க மிகவும் கஷ்டமாக உள்ளது",
        "BBMP-SAMPLE-004 (Telugu): Fallen tree blocking road": "పెద్ద చెట్టు రోడ్డుపై పడిపోయింది, వాహన రాకపోకలు నిలిచిపోయాయి, వెంటనే తొలగించాలి",
        "BBMP-SAMPLE-005 (English): Open manhole hazard": "There is an open manhole on the footpath near Koramangala 5th Block. It is uncovered and poses a serious danger to pedestrians, especially children.",
    }

    tab_voice, tab_image, tab_combined = st.tabs([
        "🎤 Voice/Text Submission",
        "📷 Image/Video Submission",
        "🔗 Combined Multimodal Submission"
    ])

    # ------------------------------------------------------------------
    # TAB 1: VOICE / TEXT SUBMISSION
    # ------------------------------------------------------------------
    with tab_voice:
        st.subheader("🎤 Voice or Regional-Language Text Grievance")
        st.markdown("Submit grievances in regional languages — audio is transcribed with Whisper and translated into English for automated severity & department triage.")

        # Preset helper
        selected_sample = st.selectbox(
            "💡 Load a Pre-configured Sample Grievance",
            list(SAMPLE_VOICE_TEXTS.keys()),
            key="tab1_sample_select"
        )
        prefill_text = SAMPLE_VOICE_TEXTS.get(selected_sample, "")

        v_c1, v_c2 = st.columns([1, 1])
        with v_c1:
            lang_label = st.selectbox("Preferred / Known Language", list(LANGUAGE_OPTIONS.keys()), key="voice_lang")
            lang_code = LANGUAGE_OPTIONS[lang_label]
            complaint_text = st.text_area(
                "Complaint Description (Voice Transcript or Typed)",
                value=prefill_text,
                placeholder=PLACEHOLDER_TEXT.get(lang_code, PLACEHOLDER_TEXT["en"]),
                height=130,
                key="voice_text"
            )
        with v_c2:
            st.markdown("**Upload Voice Note / Audio Recording**")
            audio_file = st.file_uploader(
                "Supports WAV, MP3, OGG, M4A",
                type=["wav", "mp3", "ogg", "m4a"],
                key="voice_audio"
            )
            # Sample audio button
            sample_audio_path = os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "audio", "BBMP-SAMPLE-006.wav")
            use_sample_audio = False
            if os.path.exists(sample_audio_path):
                if st.checkbox("🎵 Use Sample Audio Note (Burst Water Pipe - BBMP-SAMPLE-006)", key="chk_sample_audio"):
                    use_sample_audio = True
                    st.audio(sample_audio_path)
            if audio_file is not None:
                st.audio(audio_file)

        if st.button("🚀 Analyze & Triage Voice/Text Grievance", key="voice_submit"):
            processor = load_voice_processor()
            if processor is None:
                st.error("Multimodal modules are not available. Install FEAT-001 src modules first.")
            else:
                _t1_start = _time.time()
                with st.spinner("Processing regional language speech and grievance triage..."):
                    audio_to_process = None
                    tmp_audio_path = None
                    if audio_file is not None:
                        suffix = "." + audio_file.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(audio_file.read())
                            tmp_audio_path = tmp.name
                        audio_to_process = tmp_audio_path
                    elif use_sample_audio:
                        audio_to_process = sample_audio_path

                    try:
                        if audio_to_process:
                            result = processor.process_voice_complaint(audio_to_process)
                        else:
                            input_text = complaint_text.strip() if complaint_text else ""
                            if not input_text:
                                st.warning("Please enter a complaint description or provide an audio note.")
                                st.stop()
                            detected_lang = processor.detect_language(input_text)
                            translated = processor.translate_to_english(input_text, detected_lang)
                            result = {
                                "detected_language": detected_lang,
                                "original_text": input_text,
                                "translated_text": translated,
                                "confidence": 0.98,
                                "_text_only": True,
                            }
                    finally:
                        if tmp_audio_path and os.path.exists(tmp_audio_path):
                            try:
                                os.unlink(tmp_audio_path)
                            except OSError:
                                pass

                # Determine the English text for triage
                detected_lang_display = result.get("detected_language", "en")
                english_text = result.get("translated_text") or result.get("original_text", complaint_text or "")

                # Run through multimodal processor helper to get full triage
                mm_proc = load_multimodal_processor()
                if mm_proc:
                    full_res = mm_proc.process_complaint(text=result.get("original_text", complaint_text), ward="Bellandur")
                    severity_label = full_res["severity_label"]
                    priority_result = {
                        "priority_score": full_res["priority_score"],
                        "priority_level": full_res["priority_level"],
                        "explanation": full_res["explanation"],
                    }
                    dept_routed = full_res.get("department_routed", "BBMP Central Grievance Cell")
                    target_sla = full_res.get("target_sla", "Resolution within 3–5 Business Days")
                    suggested_cat = full_res.get("suggested_category", "General")
                    detected_lang_display = full_res.get("detected_language", detected_lang_display)
                    if full_res.get("translated_text"):
                        english_text = full_res["translated_text"]
                        result["translated_text"] = english_text
                else:
                    severity_label = "MEDIUM"
                    dept_routed = "BBMP Central Grievance Cell"
                    target_sla = "Resolution within 3–5 Business Days"
                    suggested_cat = "General"
                    priority_result = {"priority_score": 50.0, "priority_level": "MEDIUM", "explanation": ""}

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 1: Key Metrics
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                _LANG_NAMES = {'en': 'English', 'kn': 'Kannada', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu'}
                _lang_full = _LANG_NAMES.get(detected_lang_display.lower(), detected_lang_display.upper())
                _lang_display_str = f'{detected_lang_display.upper()} ({_lang_full})'
                sev_color = {"CRITICAL": "#EF4444", "HIGH": "#F97316", "MEDIUM": "#F59E0B", "LOW": "#10B981"}.get(severity_label, "#94A3B8")
                pri_color = {"CRITICAL": "#EF4444", "HIGH": "#F97316", "MEDIUM": "#F59E0B", "LOW": "#10B981"}.get(priority_result.get("priority_level", "MEDIUM"), "#94A3B8")

                with r1c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Language Identified</div><div class="metric-value" style="color:#3B82F6">{_lang_display_str}</div></div>', unsafe_allow_html=True)
                with r1c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Category</div><div class="metric-value" style="font-size:17px;color:#F8FAFC">{suggested_cat}</div></div>', unsafe_allow_html=True)
                with r1c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Severity</div><div class="metric-value" style="color:{sev_color}">{severity_label}</div></div>', unsafe_allow_html=True)
                with r1c4:
                    st.markdown(f'<div class="metric-card"><div class="metric-title">Priority Score</div><div class="metric-value" style="color:{pri_color}">{priority_result.get("priority_score", 0):.1f} / 100</div></div>', unsafe_allow_html=True)

                # Department Routing & SLA banner
                st.markdown(
                    f'<div style="background:#1E293B;border-left:5px solid #3B82F6;border-radius:6px;padding:12px 18px;margin:12px 0">'
                    f'<b>🏢 Assigned Municipal Department:</b> <span style="color:#60A5FA">{dept_routed}</span> &nbsp;|&nbsp; '
                    f'<b>⏱️ Target SLA:</b> <span style="color:{pri_color}">{target_sla}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Audio Transcription & Translation Side by Side
                if result.get("original_text"):
                    _tr_c1, _tr_c2 = st.columns(2)
                    with _tr_c1:
                        st.markdown('**📝 Original Submission**')
                        st.code(result.get('original_text', ''), language=None)
                    with _tr_c2:
                        st.markdown('**🌐 English Translation (NLP Model Output)**')
                        trans_val = result.get('translated_text')
                        if not trans_val or trans_val == result.get('original_text'):
                            if detected_lang_display != 'en':
                                trans_val = processor.translate_to_english(result.get('original_text', ''), detected_lang_display)
                            else:
                                trans_val = result.get('original_text', '')
                        st.code(trans_val, language=None)

                if priority_result.get("explanation"):
                    st.info(f"**Triage Explanation:** {priority_result['explanation']}")

                # Download JSON
                _copy_data = json.dumps({
                    'detected_language': detected_lang_display,
                    'category': suggested_cat,
                    'department_routed': dept_routed,
                    'severity': severity_label,
                    'priority_score': priority_result.get('priority_score', 0),
                    'priority_level': priority_result.get('priority_level', ''),
                    'sla_target': target_sla,
                    'original_text': result.get('original_text', complaint_text or ''),
                    'translated_text': result.get('translated_text', ''),
                    'explanation': priority_result.get('explanation', ''),
                }, indent=2, ensure_ascii=False).encode('utf-8')
                st.download_button('📋 Download Triage Result as JSON', data=_copy_data, file_name='voice_triage_result.json', mime='application/json')

                _t1_elapsed = _time.time() - _t1_start
                st.success(f'✅ Processing completed in {_t1_elapsed:.2f}s')

    # ------------------------------------------------------------------
    # TAB 2: IMAGE / VIDEO SUBMISSION
    # ------------------------------------------------------------------
    with tab_image:
        st.subheader("📷 Image / Video Complaint Inspection & Visual AI")
        st.markdown("Upload a photo or video — YOLOv8 and computer vision detect municipal issues, annotate bounding boxes, and validate authentic civic reports.")

        SAMPLE_IMAGES = {
            "-- Upload Custom Media --": None,
            "Sample 1: Large Road Pothole (BBMP-SAMPLE-007)": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-007.jpg"),
            "Sample 2: Overflowing Garbage Dumpster (BBMP-SAMPLE-008)": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-008.jpg"),
            "Sample 3: Fallen Tree Blocking Carriageway (BBMP-SAMPLE-010)": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-010.jpg"),
            "Sample 4: Realistic Bangalore Road Crater (test_pothole.jpg)": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "test_pothole.jpg"),
        }

        sel_img_preset = st.selectbox("💡 Quick Test Presets", list(SAMPLE_IMAGES.keys()), key="img_preset_select")

        media_file = st.file_uploader(
            "Upload Image or Video (JPG, PNG, MP4)",
            type=["jpg", "jpeg", "png", "mp4"],
            key="image_upload"
        )

        preset_path = SAMPLE_IMAGES.get(sel_img_preset)
        if st.button("🔍 Run Visual Inspection & Object Detection", key="image_submit"):
            target_image_path = None
            tmp_img_path = None
            media_bytes = None
            is_video = False

            if media_file is not None:
                media_bytes = media_file.getvalue()
                suffix = "." + media_file.name.split(".")[-1]
                is_video = suffix.lower() in [".mp4", ".avi", ".mov"]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(media_bytes)
                    tmp_img_path = tmp.name
                target_image_path = tmp_img_path
            elif preset_path and os.path.exists(preset_path):
                target_image_path = preset_path
                is_video = target_image_path.lower().endswith(('.mp4', '.avi', '.mov'))

            if not target_image_path:
                st.warning("Please upload an image/video file or pick a sample preset above.")
            else:
                inspector = load_visual_inspector()
                if inspector is None:
                    st.error("Visual inspection module unavailable. Please check installations.")
                else:
                    _t2_start = _time.time()
                    try:
                        with st.spinner("Analyzing image features, detecting civic anomalies, and drawing bounding boxes..."):
                            result = inspector.inspect_image(target_image_path)
                            annotated_arr = inspector.annotate_image(target_image_path, result.get("detections", []))

                        # Validity badge
                        is_valid = result.get("is_valid_complaint", False)
                        v_score = result.get("validation_score", 0.0)
                        valid_color = "#10B981" if is_valid else "#EF4444"
                        valid_label = "✅ Authentic Civic Grievance" if is_valid else "❌ Unverified / Irrelevant Scene"

                        st.markdown(
                            f'<div class="metric-card" style="margin-bottom:12px"><div class="metric-title">Verification Status</div>'
                            f'<div class="metric-value" style="color:{valid_color}">{valid_label} &nbsp; ({v_score * 100:.1f}% Confidence)</div></div>',
                            unsafe_allow_html=True
                        )

                        # Metric cards row
                        _m1, _m2, _m3, _m4 = st.columns(4)
                        category = inspector.map_detection_to_category(result.get('detections', []))
                        severity = result.get('estimated_severity', 'LOW')
                        sev_color = {'CRITICAL': '#EF4444', 'HIGH': '#F97316', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'}.get(severity, '#94A3B8')

                        with _m1:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Validation Score</div><div class="metric-value">{v_score * 100:.1f}%</div></div>', unsafe_allow_html=True)
                        with _m2:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Auto-Detected Category</div><div class="metric-value" style="font-size:18px">{category}</div></div>', unsafe_allow_html=True)
                        with _m3:
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Estimated Severity</div><div class="metric-value" style="color:{sev_color}">{severity}</div></div>', unsafe_allow_html=True)
                        with _m4:
                            _det_count = len(result.get('detections', []))
                            st.markdown(f'<div class="metric-card"><div class="metric-title">Detections Count</div><div class="metric-value">{_det_count}</div></div>', unsafe_allow_html=True)

                        # Side-by-side Visual Inspection Display
                        st.markdown("### 🖼️ Visual Object Detection & Annotation")
                        img_col1, img_col2 = st.columns(2)
                        with img_col1:
                            st.markdown("**Original Uploaded Media**")
                            if media_bytes is not None:
                                if is_video:
                                    st.video(media_bytes)
                                else:
                                    st.image(media_bytes, use_container_width=True)
                            else:
                                if is_video:
                                    st.video(target_image_path)
                                else:
                                    st.image(target_image_path, use_container_width=True)
                        with img_col2:
                            st.markdown("**AI Annotated Bounding Boxes & Civic Tags**")
                            if annotated_arr is not None:
                                st.image(annotated_arr, caption="YOLOv8 + CV Infrastructure Anomaly Annotations", use_container_width=True)
                            else:
                                st.info("No bounding boxes to render.")

                        # Civic issues as colored tags
                        civic_issues = result.get("civic_issues", [])
                        if civic_issues:
                            st.subheader("Identified Civic Issue Flags")
                            _ISSUE_ICONS = {'pothole': '🚧', 'garbage': '🗑️', 'tree': '🌳', 'fallen': '🌳', 'water': '💧', 'road': '🛣️', 'drain': '🌊'}
                            _tags_html = ''.join(
                                f'<span style="background:#1E293B;border:1px solid #3B82F6;border-radius:8px;padding:5px 14px;margin:4px;display:inline-block;color:#93C5FD">'
                                f'{_ISSUE_ICONS.get(issue.lower().split()[0], "🏙️")} {issue.upper()}</span>'
                                for issue in civic_issues
                            )
                            st.markdown(f'<div style="margin:8px 0">{_tags_html}</div>', unsafe_allow_html=True)

                        # Detections table
                        detections = result.get("detections", [])
                        if detections:
                            st.subheader("Detected Objects & Bounding Coordinates")
                            det_df = pd.DataFrame(detections)
                            if 'confidence' in det_df.columns:
                                det_df['confidence'] = det_df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
                            if 'bbox' in det_df.columns:
                                det_df['bbox'] = det_df['bbox'].apply(lambda x: str(x))
                            st.dataframe(det_df.reset_index(drop=True), use_container_width=True)

                        _t2_elapsed = _time.time() - _t2_start
                        st.success(f'✅ Visual analysis completed in {_t2_elapsed:.2f}s')
                    finally:
                        if tmp_img_path and os.path.exists(tmp_img_path):
                            try:
                                os.unlink(tmp_img_path)
                            except OSError:
                                pass

    # ------------------------------------------------------------------
    # TAB 3: COMBINED MULTIMODAL SUBMISSION
    # ------------------------------------------------------------------
    with tab_combined:
        st.subheader("🔗 Combined Multimodal Grievance Triage Pipeline")
        st.markdown("Fuse text, regional audio notes, and photos/videos into a single composite triage decision with cross-modal validation, SLA deadlines, and municipal department assignment.")

        COMBINED_PRESETS = {
            "-- Custom Multimodal Input --": None,
            "Scenario 1: Kannada Audio + Pothole Photo (Bellandur Ward)": {
                "text": "ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಹೊಂಡ ಇದೆ, ವಾಹನಗಳಿಗೆ ಹಾನಿ ಆಗುತ್ತಿದೆ",
                "audio": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "audio", "BBMP-SAMPLE-006.wav"),
                "image": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-007.jpg"),
                "ward": "Bellandur",
            },
            "Scenario 2: Hindi Text + Garbage Dump Photo (Horamavu Ward)": {
                "text": "मुख्य सड़क के पास कचरे का ढेर बहुत बड़ा हो गया है और बदबू आ रही है",
                "audio": None,
                "image": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-008.jpg"),
                "ward": "Horamavu",
            },
            "Scenario 3: Telugu Text + Fallen Tree Roadblock (Jayanagar Ward)": {
                "text": "పెద్ద చెట్టు రోడ్డుపై పడిపోయింది, వాహన రాకపోకలు నిలిచిపోయాయి, వెంటనే తొలగించాలి",
                "audio": None,
                "image": os.path.join(PROJECT_ROOT, "data", "sample_multimodal", "images", "BBMP-SAMPLE-010.jpg"),
                "ward": "Jayanagar",
            },
            "Scenario 4: English Open Manhole Emergency (Koramangala Ward)": {
                "text": "There is an open manhole on the footpath near Koramangala 5th Block. It is uncovered and poses a serious danger to pedestrians, especially children.",
                "audio": None,
                "image": None,
                "ward": "Koramangala",
            },
        }

        comb_preset_choice = st.selectbox("💡 Pre-configured Multimodal Demonstration Scenarios", list(COMBINED_PRESETS.keys()), key="comb_preset_select")
        active_preset = COMBINED_PRESETS.get(comb_preset_choice) or {}

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            combined_text = st.text_area(
                "Complaint Description (English or Regional Indian Language)",
                value=active_preset.get("text", ""),
                placeholder="Describe the municipal grievance...",
                height=110,
                key="combined_text"
            )
            combined_audio = st.file_uploader("Upload Audio Note (optional)", type=["wav", "mp3", "ogg", "m4a"], key="combined_audio")
            preset_audio_path = active_preset.get("audio")
            if preset_audio_path and os.path.exists(preset_audio_path):
                st.info(f"🎵 Scenario Audio Attached: `{os.path.basename(preset_audio_path)}`")
                st.audio(preset_audio_path)

        with c_col2:
            combined_image = st.file_uploader("Upload Photo / Video (optional)", type=["jpg", "jpeg", "png", "mp4"], key="combined_image")
            preset_image_path = active_preset.get("image")
            if preset_image_path and os.path.exists(preset_image_path):
                st.info(f"📷 Scenario Image Attached: `{os.path.basename(preset_image_path)}`")
                st.image(preset_image_path, width=220)

            ward_options = sorted(list(df["Ward_Name_Clean"].dropna().unique()))
            default_ward_idx = 0
            preset_ward = active_preset.get("ward", "Bellandur")
            for idx, w in enumerate(ward_options):
                if preset_ward.lower() in w.lower():
                    default_ward_idx = idx
                    break
            combined_ward = st.selectbox("Ward Location", ward_options, index=default_ward_idx, key="combined_ward")

        c_col3, c_col4 = st.columns(2)
        with c_col3:
            combined_date = st.date_input("Grievance Incurred Date", key="combined_date")
        with c_col4:
            category_options = sorted(list(df["Category_Clean"].dropna().unique()))
            combined_category = st.selectbox("Category Override (optional)", ["Auto-detect"] + category_options, key="combined_category")

        if st.button("🚀 Process End-to-End Multimodal Grievance", key="combined_submit"):
            mm_processor = load_multimodal_processor()
            if mm_processor is None:
                st.error("Multimodal modules are not available. Install FEAT-001 src modules first.")
            else:
                audio_path = None
                image_path = None
                tmp_files = []
                _t3_start = _time.time()

                try:
                    if combined_audio is not None:
                        suffix = "." + combined_audio.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(combined_audio.read())
                            audio_path = tmp.name
                            tmp_files.append(audio_path)
                    elif preset_audio_path and os.path.exists(preset_audio_path):
                        audio_path = preset_audio_path

                    if combined_image is not None:
                        suffix = "." + combined_image.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(combined_image.read())
                            image_path = tmp.name
                            tmp_files.append(image_path)
                    elif preset_image_path and os.path.exists(preset_image_path):
                        image_path = preset_image_path

                    annotated_display = None
                    with st.spinner("Executing end-to-end multimodal fusion pipeline..."):
                        result = mm_processor.process_complaint(
                            text=combined_text,
                            audio_path=audio_path,
                            image_path=image_path,
                            ward=combined_ward,
                            complaint_date=str(combined_date),
                        )
                        if image_path and os.path.exists(image_path):
                            inspector = load_visual_inspector()
                            if inspector:
                                annotated_display = inspector.annotate_image(
                                    image_path,
                                    (result.get("vision_result") or {}).get("detections", [])
                                )

                    # Results Display
                    st.markdown('---')

                    # Section 1: Inputs Received
                    st.markdown('### 📥 Inputs Received & Registered')
                    _inp_c1, _inp_c2, _inp_c3, _inp_c4 = st.columns(4)
                    _has_text = bool(combined_text.strip())
                    _has_audio = audio_path is not None
                    _has_image = image_path is not None
                    _inp_c1.markdown(f"{'✅' if _has_text else '➖'} **Text:** {'Provided' if _has_text else 'None'}")
                    _inp_c2.markdown(f"{'✅' if _has_audio else '➖'} **Audio:** {'Attached' if _has_audio else 'None'}")
                    _inp_c3.markdown(f"{'✅' if _has_image else '➖'} **Vision:** {'Uploaded' if _has_image else 'None'}")
                    _inp_c4.markdown(f"📍 **Ward:** {combined_ward}")

                    # Section 2: Intermediate Analysis
                    st.markdown('### 🔍 Intermediate Multimodal Modality Analysis')
                    _voice_result = result.get('voice_result') or {}
                    _vision_result = result.get('vision_result') or {}

                    _exp_col1, _exp_col2 = st.columns(2)

                    with _exp_col1:
                        with st.expander('🎤 Voice & Regional NLP Analysis', expanded=True):
                            st.markdown(f"**Identified Language:** `{result.get('language_name', 'English')} ({result.get('detected_language', 'en').upper()})`")
                            if result.get('translated_text'):
                                st.markdown("**Original Submission:**")
                                st.code(result.get('original_text', ''), language=None)
                                st.markdown("**English Translation:**")
                                st.code(result.get('translated_text', ''), language=None)
                            else:
                                st.markdown(f"**Grievance Text:** `{result.get('processed_text', 'N/A')}`")

                    with _exp_col2:
                        with st.expander('📷 Computer Vision Inspection', expanded=True):
                            if _vision_result:
                                _vis_cols = st.columns(3)
                                _vis_cols[0].metric('Validation Score', f"{_vision_result.get('validation_score', 0) * 100:.1f}%")
                                _vis_cols[1].metric('Detections', len(_vision_result.get('detections', [])))
                                _vis_cols[2].metric('Visual Severity', result.get('visual_severity', 'LOW'))

                                civ_issues = _vision_result.get('civic_issues', [])
                                if civ_issues:
                                    st.markdown(f"**Civic Flags Detected:** `{', '.join(civ_issues)}`")

                                # Render annotated image if available
                                if annotated_display is not None:
                                    st.image(annotated_display, caption="AI Annotated Detection Overlays", use_container_width=True)
                                elif image_path and os.path.exists(image_path):
                                    inspector = load_visual_inspector()
                                    if inspector:
                                        annotated_display = inspector.annotate_image(image_path, _vision_result.get('detections', []))
                                        if annotated_display is not None:
                                            st.image(annotated_display, caption="AI Annotated Detection Overlays", use_container_width=True)
                            else:
                                st.info("No visual media provided. Text/voice heuristics utilized.")

                    # Section 3: Final Priority Score & Municipal Action Card
                    st.markdown('### 📊 Final Composite Priority Score & Dispatch Decision')
                    final_score = result.get('priority_score', 0)
                    final_level = result.get('priority_level', 'N/A')
                    final_sev = result.get('severity_label', 'UNKNOWN')
                    _fs_color = {'CRITICAL': '#EF4444', 'HIGH': '#F97316', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'}.get(final_level, '#94A3B8')

                    st.markdown(
                        f'<div class="metric-card" style="text-align:center;padding:25px;margin-bottom:15px">'
                        f'<div class="metric-title">COMPOSITE RESOLUTION-PRIORITY SCORE</div>'
                        f'<div class="metric-value" style="font-size:52px;color:{_fs_color}">{final_score:.1f}</div>'
                        f'<div style="color:#94A3B8;font-size:16px">Scale 0–100 &nbsp;|&nbsp; Priority Level: <span style="color:{_fs_color};font-weight:bold">{final_level}</span> &nbsp;|&nbsp; Severity: <span style="color:{_fs_color};font-weight:bold">{final_sev}</span></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    _detail_c1, _detail_c2, _detail_c3, _detail_c4 = st.columns(4)
                    with _detail_c1:
                        st.markdown(f'<div class="metric-card"><div class="metric-title">Complaint ID</div><div class="metric-value" style="font-size:15px;color:#60A5FA">{result.get("complaint_id", "N/A")}</div></div>', unsafe_allow_html=True)
                    with _detail_c2:
                        _rv_color = "#F59E0B" if result.get('requires_human_review') else "#10B981"
                        _rv_label = "⚠️ Human Review" if result.get('requires_human_review') else "✅ Verified Auto-Dispatch"
                        st.markdown(f'<div class="metric-card"><div class="metric-title">Verification Status</div><div class="metric-value" style="color:{_rv_color};font-size:15px">{_rv_label}</div></div>', unsafe_allow_html=True)
                    with _detail_c3:
                        st.markdown(f'<div class="metric-card"><div class="metric-title">Assigned Category</div><div class="metric-value" style="font-size:15px;color:#F8FAFC">{result.get("suggested_category", "General")}</div></div>', unsafe_allow_html=True)
                    with _detail_c4:
                        st.markdown(f'<div class="metric-card"><div class="metric-title">Target SLA Resolution</div><div class="metric-value" style="color:{_fs_color};font-size:15px">{result.get("target_sla", "N/A")}</div></div>', unsafe_allow_html=True)

                    # Department assignment card
                    st.markdown(
                        f'<div style="background:#1E293B;border-left:5px solid #10B981;border-radius:6px;padding:14px 18px;margin:15px 0">'
                        f'<b>🏢 Routed Municipal Authority:</b> <span style="color:#34D399;font-size:16px;font-weight:bold">{result.get("department_routed", "BBMP Central Office")}</span><br>'
                        f'<b>📋 Operational Recommendation:</b> {result.get("explanation", "")}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Download JSON
                    _full_json = json.dumps(result, indent=2, default=str, ensure_ascii=False).encode('utf-8')
                    st.download_button('📥 Download Full Multimodal Triage Record as JSON', data=_full_json, file_name=f"{result.get('complaint_id', 'complaint')}_triage.json", mime='application/json')

                    _t3_elapsed = _time.time() - _t3_start
                    st.success(f'✅ End-to-end multimodal complaint triage successfully executed in {_t3_elapsed:.2f}s')
                finally:
                    for f in tmp_files:
                        try:
                            os.unlink(f)
                        except OSError:
                            pass

