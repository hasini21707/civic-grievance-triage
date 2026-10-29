# Civic Grievance Triage and Resolution-Priority Scoring: A Composite, Multimodal Machine-Learning Framework for Municipal Complaint Dispatch Ranking

An end-to-end Data Science and Multimodal Artificial Intelligence system designed to prioritize civic complaints according to actual urgency, historical ward recurrence risk, and complaint aging—replacing static First-In, First-Out (FIFO) complaint handling with dynamic, data-driven municipal dispatch ranking.

---

## 📌 Problem Statement

Municipal corporations receive tens of thousands of citizen grievances monthly across roads, sanitation, electrical utilities, drainage, and parks. In traditional grievance portals, complaints are processed chronologically (FIFO). Consequently:
- Life-threatening emergencies (e.g., exposed live electrical wires, collapsed drainage culverts, major roadway sinkholes) are queued behind minor non-hazardous requests (e.g., routine park pruning or dim streetlights).
- Chronic municipal infrastructure failure hotspots are treated identically to isolated one-off incidents.
- Citizens communicating in regional languages or submitting photos/audio recordings are frequently underserved due to monolingual text-only triage systems.

This project resolves these operational bottlenecks by providing an automated, multimodal decision-support framework that validates, categorizes, and scores every grievance on a **0–100 Resolution-Priority Scale**.

---

## 🔬 Core Innovations & Scientific Approach

1. **Multimodal Grievance Ingestion**:
   - **Regional Voice Notes & Text**: OpenAI Whisper ASR + deterministic Unicode script detection + multi-tier neural translation across **Kannada, Devanagari/Hindi, Tamil, Telugu, and English**.
   - **Computer Vision Inspection**: YOLOv8 object detection combined with OpenCV municipal anomaly heuristics (asphalt contour cavity analysis, HSV saturation variance, edge density, and foliage masking) with automated bounding box annotation.
2. **Composite Resolution-Priority Scoring Formulation**:
   $$\text{Priority Score} = 0.50 \times \text{Severity} + 0.30 \times \text{Recurrence Risk} + 0.20 \times \text{Complaint Aging}$$
   - **Severity (50%)**: Derived via Natural Language Processing (TF-IDF + Naive Bayes) calibrated with municipal domain hazard keyword heuristics.
   - **Ward Recurrence Risk (30%)**: Unsupervised **K-Means clustering** across 199 municipal wards to detect chronic infrastructure bottlenecks.
   - **Complaint Aging (20%)**: Dynamic temporal aging curve ensuring lower-severity grievances do not suffer indefinite starvation.
3. **Automated Department Routing & SLA Deadlines**:
   - Instant routing to responsible municipal agencies (e.g., *Roads & Infrastructure*, *Solid Waste Management*, *Storm Water Drain*, *Forest/Trees Department*, *Streetlights/BESCOM*).
   - Time-bound Target Service Level Agreements (SLAs) ranging from 4–12 hours for critical emergencies to 3–5 business days for routine issues.

---

## 📊 Dataset Ingestion & Empirical Profiling

The model training and empirical evaluations are based on real-world municipal grievance records sourced from OpenCity for Bengaluru (**BBMP Grievances 2020–2025**):

- **Total Combined Records**: `766,648`
- **Yearly Record Breakdown**:
  - `2020`: 91,620
  - `2021`: 103,504
  - `2022`: 118,394
  - `2023`: 119,140
  - `2024`: 207,016
  - `2025`: 126,974
- **Original Dataset Attributes**: `['Complaint ID', 'Category', 'Sub Category', 'Grievance Date', 'Ward Name', 'Grievance Status', 'Staff Remarks', 'Staff Name']`
- **Missing Value Profile**: Core triage fields (`Category`, `Sub Category`, `Grievance Date`, `Ward Name`) are **100% complete**. Only operational fields have minimal missing values: `Grievance Status` (0.010%), `Staff Remarks` (0.007%), `Staff Name` (0.007%).
- **Geographic Coverage**: `199` municipal wards
- **Taxonomy**: `32` primary civic categories and `181` specialized sub-categories
- **Temporal Range**: February 8, 2020 through June 19, 2025

### Dataset Availability
The complete consolidated processed dataset (`cleaned_grievances.csv`) exceeds GitHub's 100 MB single-file limit. Raw annual datasets can be downloaded directly from the [OpenCity Data Portal](https://data.opencity.in/) and placed in `data/raw/` to reproduce the preprocessing pipeline.

---

## 🏗 Modular Project Architecture

```text
civic_grievance_project/
├── data/
│   ├── raw/                             # 2020-2025 raw municipal grievance CSV files
│   ├── processed/                       # Cleaned dataset, ward recurrence profiles, dispatch queue
│   └── sample_multimodal/               # Multi-language audio recordings and sample test images
│       ├── audio/                       # Kannada (BBMP-SAMPLE-006.wav), Hindi (BBMP-SAMPLE-009.wav)
│       └── images/                      # Pothole, garbage, fallen tree, and real road crater images
├── models/                              # Serialized ML artifacts (.joblib)
│   ├── tfidf_vectorizer.joblib          # Text feature extraction
│   ├── naive_bayes_model.joblib         # Severity classifier
│   └── kmeans_recurrence_model.joblib   # 199-Ward K-Means risk cluster model
├── notebooks/                           # Experimental exploratory notebooks
│   ├── 01_EDA.ipynb                     # Exploratory Data Analysis & distribution profiling
│   ├── 02_Preprocessing.ipynb           # Cleaning, imputation, and normalization pipeline
│   ├── 03_Severity_Classification.ipynb # ML benchmark (Naive Bayes, LR, Decision Trees, KNN)
│   └── 04_KMeans_Recurrence.ipynb       # K-Means clustering, silhouette scores, elbow curve
├── src/                                 # Production core engine modules
│   ├── __init__.py
│   ├── data_processing.py               # Robust multi-year ingestion, cleaning, deduplication
│   ├── feature_engineering.py           # Severity labeling, keyword hazards, aging calculations
│   ├── severity_model.py                # Text vectorization and multi-model classification
│   ├── recurrence_model.py              # K-Means ward clustering and spatial risk scoring
│   ├── priority_score.py                # Mathematical composite priority scoring engine
│   ├── voice_nlp_processor.py           # Whisper ASR, Unicode script detection, neural translation
│   ├── visual_inspector.py              # YOLOv8 + OpenCV anomaly detection & image annotation
│   └── multimodal_complaint.py          # Unified multimodal complaint fusion & triage dispatch
├── dashboard/
│   └── app.py                           # Interactive 7-Page Streamlit Web Application
├── tests/                               # Comprehensive Automated Pytest Suite
│   ├── test_data_processing.py          # Unit tests for loading and cleaning
│   ├── test_severity_model.py           # Unit tests for TF-IDF & classification
│   ├── test_recurrence_model.py         # Unit tests for ward clustering & scoring
│   └── test_multimodal.py               # Unit tests for Voice, Vision, Fusion & SLA routing
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 🌐 Multimodal AI Subsystems

### 1. 🎤 Multilingual Voice & NLP Pipeline (`src/voice_nlp_processor.py`)
- **Speech Recognition (ASR)**: Uses OpenAI Whisper to transcribe audio files (`.wav`, `.mp3`, `.ogg`, `.m4a`).
- **Deterministic Script Detection**: Employs Unicode block boundary mapping (Kannada `\u0C80-\u0CFF`, Devanagari `\u0900-\u097F`, Tamil `\u0B80-\u0BFF`, Telugu `\u0C00-\u0C7F`) to achieve 100% accuracy on Indic scripts.
- **Multi-Tier Translation Engine**:
  1. *In-Memory Exact Phrase Cache* for sub-millisecond response on common municipal phrases.
  2. *Online Neural Translation Endpoint* with a strict 4-second timeout.
  3. *Cached HuggingFace MarianMT Models* (`Helsinki-NLP/opus-mt-{src}-en`).
  4. *Curated Indic Civic Lexicon Fallback* mapping regional vocabulary directly to standard municipal terms.
- **Language Coverage**:
  | Language | Code | Native Script | Primary Model / Pipeline |
  | :--- | :---: | :---: | :--- |
  | **Kannada** | `kn` | ಕನ್ನಡ | Whisper ASR + Opus-MT-kn-en / Civic Lexicon |
  | **Hindi** | `hi` | हिन्दी | Whisper ASR + Opus-MT-hi-en / Civic Lexicon |
  | **Tamil** | `ta` | தமிழ் | Whisper ASR + Opus-MT-ta-en / Civic Lexicon |
  | **Telugu** | `te` | తెలుగు | Whisper ASR + Neural Endpoint / Civic Lexicon |
  | **English** | `en` | Latin | Direct Passthrough + Keyword Extraction |

### 2. 👁️ Computer Vision Inspection & Annotation (`src/visual_inspector.py`)
- **YOLOv8 Detection**: Inspects uploaded images or video keyframes (`.jpg`, `.png`, `.mp4`) using `yolov8n.pt`.
- **OpenCV Municipal Anomaly Detection**:
  - *Pothole / Road Cavities*: Adaptive Gaussian thresholding, contour area geometric filtering, and aspect ratio validation on asphalt surfaces.
  - *Garbage Dumps*: Localized HSV color saturation variance and edge density quantification.
  - *Fallen Trees / Roadblocks*: Foliage green-hue segmentation masks and horizontal bounding span analysis.
  - *Flooding / Waterlogging*: Surface reflection ratio and dark pool edge continuity detection.
- **Visual Annotation Engine**: `annotate_image()` generates color-coded bounding boxes, confidence tags, and category labels overlaying the media.
- **Fraud & Irrelevance Filter**: Assesses image credibility via `validation_score` (0.0 to 1.0) and flags non-civic uploads.

### 3. 🔗 Multimodal Complaint Fusion (`src/multimodal_complaint.py`)
- Cross-validates text, audio, and visual modalities into a single operational complaint schema.
- Dynamically assigns municipal departments and SLA targets.
- Flags uncertain or contradictory cases for **Human Review** while enabling **Verified Auto-Dispatch** for high-confidence grievances.
- Exports full audit-ready JSON triage records.

---

## 🖥️ Interactive 7-Page Streamlit Web Dashboard (`dashboard/app.py`)

The system features an interactive, dark-themed Streamlit application:

1. **📊 Overview**: High-level municipal KPIs, complaint volume distributions, category breakdowns, monthly seasonality charts, and resolution status metrics across 766K+ records.
2. **🔎 Complaint Analysis**: Interactive data explorer with dynamic multi-attribute filtering (year, ward, category, status).
3. **📋 Priority Queue**: Live operational dispatch queue ranked descending by Priority Score with 1-click CSV export for municipal zonal officers.
4. **🗺️ Ward Recurrence Risk**: Interactive **Folium OpenStreetMap** visualizing 199 municipal wards color-coded by K-Means risk tier (High, Medium, Low) with ward-level breakdown tables.
5. **⚡ Complaint Predictor**: Real-time triage form with instant severity prediction, composite priority score calculation, and plain-language explanation cards.
6. **📈 Model Performance**: Comparative evaluation matrix across multiple machine learning models (Naive Bayes, Logistic Regression, Decision Tree, KNN) detailing Precision, Recall, and F1-Scores.
7. **🌐 Multimodal Submission**:
   - **Tab 1 (Voice & NLP)**: Audio recording upload or preset selection, audio playback, original text and English translation side-by-side, department routing, and SLA targets.
   - **Tab 2 (Image & Video)**: Media upload or preset selection, side-by-side original vs AI-annotated bounding boxes, verification confidence, estimated visual severity, and coordinate detection tables.
   - **Tab 3 (Combined Multimodal)**: End-to-end multi-scenario testing combining regional voice/text, photos/videos, ward location, and date into a single composite priority score with JSON export.

---

## 🧪 Comprehensive Automated Test Suite

The project includes an automated test suite executed via `pytest`:
- **`tests/test_data_processing.py`**: Verifies data loading, schema integrity, deduplication, and cleaning pipelines.
- **`tests/test_severity_model.py`**: Validates TF-IDF vectorization, model serialization, and multi-class severity predictions.
- **`tests/test_recurrence_model.py`**: Validates K-Means clustering, silhouette computation, and composite priority scoring calculations.
- **`tests/test_multimodal.py`**: Tests Whisper transcription, Indic script detection, multi-tier translation, YOLOv8 visual inspection, bounding box rendering, department routing, and end-to-end complaint fusion.

All **21 automated tests pass cleanly**:
```bash
python -m pytest tests/ -v
# Output: 21 passed in 45s
```

---

## 🚀 Installation & Usage

### 1. Prerequisites
- **Python**: Version `3.10` or higher
- **Operating System**: Windows / Linux / macOS

### 2. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/<your-username>/civic-grievance-composite-multimodal-triage.git
cd civic-grievance-composite-multimodal-triage
pip install -r requirements.txt
```

### 3. Launch the Web Dashboard
```bash
streamlit run dashboard/app.py
```
Access the application in your browser at `http://localhost:8501`.

### 4. Run Automated Tests
```bash
python -m pytest tests/ -v
```

---

## 📈 Milestone Development Roadmap

- [x] **Stage 1 (10%)**: Dataset Ingestion, quality profiling, and schema validation across 766,648 municipal records.
- [x] **Stage 2 (20%)**: Data Preprocessing, text normalization, missing value imputation, and deduplication.
- [x] **Stage 3 (30%)**: Feature Engineering, severity labeling, critical domain keywords, and complaint aging features.
- [x] **Stage 4 (40%)**: Supervised ML Severity Classification comparing Naive Bayes, Logistic Regression, Decision Trees, and KNN.
- [x] **Stage 5 (50%)**: Unsupervised K-Means Ward Recurrence Risk Clustering and Folium geospatial mapping.
- [x] **Stage 6 (60%)**: Multimodal AI Processing (Whisper regional voice ASR, Indic NLP translation, YOLOv8 + OpenCV visual defect detection, Page 7 Streamlit multimodal UI, and test suite).
- [ ] **Stage 7 (70%)**: Configurable Priority Scoring with dynamic sensitivity weight sliders in the dashboard.
- [ ] **Stage 8 (80%)**: Baseline Comparative Evaluation quantifying SLA breach reduction vs. traditional FIFO dispatching.
- [ ] **Stage 9 (90%)**: FastAPI REST Endpoints for external municipal portal integration.
- [ ] **Stage 10 (100%)**: Persistent Database Integration (SQLite/PostgreSQL) and production containerization.

---

## 📜 License & Acknowledgments

- **Dataset Source**: Published by [OpenCity.in](https://data.opencity.in/) for municipal grievance transparency.
- **Pretrained Models**: OpenAI Whisper (`openai/whisper-tiny`), Ultralytics YOLOv8 (`yolov8n`), and MarianMT (`Helsinki-NLP`).
