# CIVIC GRIEVANCE TRIAGE & RESOLUTION-PRIORITY SCORING SYSTEM

An operational municipal decision-support system designed to prioritize civic complaints according to actual urgency, recurrence risk, and complaint aging rather than simply resolving complaints in FIFO (First-In, First-Out) order.

---

## 📌 Problem Statement

Indian municipal bodies (such as BBMP in Bengaluru) receive thousands of civic complaints daily regarding potholes, garbage accumulation, water leakage, streetlights, drainage overflows, and road damage. Currently, complaints are frequently processed in the order received (FIFO) rather than by urgency. A simple streetlight outage reported early might take precedence over a major drainage collapse or exposed live wire reported later.

This project solves this operational challenge by providing an automated triage system that generates an actionable composite Priority Score for every incoming grievance.

---

## 🔬 Research Gap & Novel Innovation

Traditional systems treat civic grievance analysis as a generic NLP complaint classification problem (e.g., mapping text to "Garbage" vs. "Roads").

**Our System Innovation**:
Combines **Complaint Severity** (via NLP ML Classification) + **Location-based Recurrence Risk** (via K-Means Ward Clustering) + **Complaint Aging** into a single dynamic operational **Priority Score (0–100)** and ranked **Daily Dispatch Worklist**.

---

## 📊 Dataset Ingestion & Empirical Profiling

The system is built and evaluated on real, freely available municipal grievance data from OpenCity for Bengaluru (**BBMP Grievances 2020–2025**):

- **Total Combined Records**: `766,648`
- **Yearly File Breakdown**:
  - `2020`: 91,620
  - `2021`: 103,504
  - `2022`: 118,394
  - `2023`: 119,140
  - `2024`: 207,016
  - `2025`: 126,974
- **Original Dataset Columns**: `['Complaint ID', 'Category', 'Sub Category', 'Grievance Date', 'Ward Name', 'Grievance Status', 'Staff Remarks', 'Staff Name']`
- **Missing Values**: `Grievance Status` (74 missing, 0.010%), `Staff Remarks` (54 missing, 0.007%), `Staff Name` (54 missing, 0.007%). Core fields are 100% complete.
- **Unique Wards**: `199` BBMP municipal wards
- **Unique Categories**: `32`
- **Unique Sub-Categories**: `181`
- **Date Range**: `2020-02-08 01:01:00` to `2025-06-19 12:59:00`

---

## 🏗 Project Architecture (50% Milestone + Streamlit Web Dashboard)

```
civic_grievance_project/
├── data/
│   ├── raw/                        # 2020-2025 raw BBMP CSV files (766,648 records)
│   └── processed/                  # Cleaned CSV, ward recurrence risk, dispatch queue
├── models/                         # Serialized models (TF-IDF vectorizer, Naive Bayes, K-Means)
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb      # Data cleaning pipeline
│   ├── 03_Severity_Classification.ipynb  # ML Severity classification comparison
│   └── 04_KMeans_Recurrence.ipynb  # Ward recurrence K-Means clustering
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Data loading, cleaning, quality reporting
│   ├── feature_engineering.py      # Keyword flags, derived severity, aging features
│   ├── severity_model.py           # TF-IDF + Naive Bayes, LR, DT, KNN classification
│   ├── recurrence_model.py         # K-Means ward recurrence risk clustering
│   └── priority_score.py           # Composite priority scoring & daily dispatch queue
├── dashboard/
│   └── app.py                      # 🖥️ Interactive 6-Page Streamlit Web App
├── tests/
│   ├── test_data_processing.py     # Pytest suite for data loading & cleaning
│   ├── test_severity_model.py      # Pytest suite for severity classification
│   └── test_recurrence_model.py    # Pytest suite for ward recurrence & priority scoring
├── requirements.txt                # Dependency manifest
└── README.md                       # Complete documentation
```

---

## 🌐 Multimodal AI Integration (Voice + Vision)

### 🎤 Voice Pipeline

Whisper ASR → language detection (langdetect) → Helsinki-NLP translation → IndicBERT-compatible text → TF-IDF + Naive Bayes severity classifier

Citizens can submit complaints as audio recordings in any supported Indian regional language. The pipeline automatically transcribes the audio using OpenAI Whisper, detects the source language, translates to English via Helsinki-NLP MarianMT models, and routes the translated text through the existing severity classification and priority scoring pipeline.

### 👁️ Vision Pipeline

Image/video input → YOLOv8n object detection → COCO→BBMP category mapping → confidence-based severity estimation → composite priority score

Photos or videos submitted alongside complaints are analyzed using YOLOv8n. Detected objects (potholes, garbage, fallen trees, open drains) are mapped to BBMP complaint categories. Confidence scores drive severity estimation, which feeds directly into the composite priority scoring formula.

### 🗣️ Supported Languages

| Language | Code | Script | Translation Model |
|----------|------|--------|-------------------|
| Kannada | kn | ಕನ್ನಡ | Helsinki-NLP/opus-mt-kn-en |
| Hindi | hi | हिन्दी | Helsinki-NLP/opus-mt-hi-en |
| Tamil | ta | தமிழ் | Helsinki-NLP/opus-mt-ta-en |
| Telugu | te | తెలుగు | Helsinki-NLP/opus-mt-te-en |
| English | en | Latin | N/A |

### 📦 Dataset References

- **Voice training data**: [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) — multilingual Indian speech corpus
- **Pothole detection**: [michelpf/dataset-pothole](https://huggingface.co/datasets/michelpf/dataset-pothole) — annotated pothole images
- **Garbage detection**: [TACO Dataset](http://taco-dataset.github.io/) — trash annotations in context
- **Indian language NLP model**: [ai4bharat/indic-bert](https://huggingface.co/ai4bharat/indic-bert) — IndicBERT for regional language NLP

### ⚙️ Installation

```bash
pip install openai-whisper==20231117 transformers==4.44.2 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 ultralytics==8.2.87 langdetect==1.0.9 Pillow==10.4.0 opencv-python==4.10.0.84 sentencepiece==0.2.0 sacremoses==0.1.1
```

---

## 🖥️ Streamlit Web Frontend Features (`dashboard/app.py`)
Run the Web App locally:
```bash
streamlit run dashboard/app.py
```

### 7 Interactive Dashboard Pages:
1. **Overview**: Key KPI metrics, grievance category bar charts, top ward charts, monthly trends, and status breakdown.
2. **Complaint Analysis**: Interactive filterable grievance dataset browser.
3. **Priority Queue**: Dynamic daily dispatch worklist sorted descending by Priority Score with CSV export.
4. **Ward Recurrence Risk**: K-Means risk grouping and interactive **Folium OpenStreetMap** ward risk map.
5. **Complaint Predictor**: Live triage predictor form with instant severity classification, priority score calculation, and plain-language explainer.
6. **Model Performance**: Performance metrics comparison table across Naive Bayes, Logistic Regression, Decision Tree, and KNN.
7. **Multimodal Submission**: Voice/text submission with regional language detection and translation, image/video inspection with YOLOv8 object detection, and combined multimodal complaint processing.

---

## 🛠 Software Requirements

- Python 3.10+
- Pandas, NumPy, Scikit-learn, Scipy
- Matplotlib, Plotly, Joblib, Folium, Streamlit
- Pytest, Jupyter

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Dashboard App
```bash
streamlit run dashboard/app.py
```

### 3. Run Pipeline Execution Script
```bash
python scratch/run_pipeline.py
```

### 4. Execute Pytest Unit Test Suite
```bash
python -m pytest tests/ -v
```

---

## 📈 Milestone Roadmap

- [x] **Stage 1 (10%)**: Dataset Ingestion, Inspection & Quality Profiling
- [x] **Stage 2 (20%)**: Data Preprocessing, Cleaning & Deduplication
- [x] **Stage 3 (30%)**: Feature Engineering & Derived Severity Labeling
- [x] **Stage 4 (40%)**: Machine Learning Severity Classification (TF-IDF + Naive Bayes, LR, DT, KNN)
- [x] **Stage 5 (50%)**: Ward Recurrence Risk Clustering (K-Means) + 🖥️ **Streamlit Frontend Dashboard**
- [x] **Stage 6 (60%)**: Advanced Multimodal AI Integration — Voice/NLP (Whisper + IndicBERT + Helsinki-NLP) + Vision (YOLOv8 + ViT) + 🌐 **Page 7 Multimodal Submission Dashboard**
  - [x] Regional language NLP (Kannada, Hindi, Tamil, Telugu) with automatic translation
  - [x] Voice/audio complaint submission via OpenAI Whisper ASR
  - [x] Image/video inspection via YOLOv8n object detection
  - [x] Automated civic issue validation (pothole depth, garbage volume, fallen trees)
  - [x] Combined multimodal complaint processing pipeline
- [ ] **Stage 7 (70%)**: Configurable Weight Matrix & Multi-Factor Priority Tuning
- [ ] **Stage 8 (80%)**: Empirical Baseline Comparison (FIFO vs Proposed Composite System)
- [ ] **Stage 9 (90%)**: FastAPI Backend Integration & OpenAPI REST Endpoints (`POST /predict`, `GET /wards/risk`)
- [ ] **Stage 10 (100%)**: SQLite Database Ingestion Pipeline & Real-Time Ready Municipal Integration
