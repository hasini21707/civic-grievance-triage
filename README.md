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

## 🏗 Project Architecture

The Civic Grievance Triage and Resolution-Priority Scoring System is designed as a modular application that combines municipal complaint data processing, machine learning, ward-level recurrence analysis, priority scoring, and an interactive Streamlit dashboard.

The architecture separates data, model development, application logic, and the user interface to make the system easier to maintain and extend.

```text
civic-grievance-triage/
│
├── data/
│   ├── raw/                         # Original municipal grievance datasets
│   └── processed/                   # Cleaned data and derived outputs
│
├── models/                          # Saved machine learning models
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb       # Data cleaning and preparation
│   ├── 03_Severity_Classification.ipynb
│   └── 04_KMeans_Recurrence.ipynb   # Ward recurrence analysis
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py           # Data loading and cleaning
│   ├── feature_engineering.py       # Severity and aging features
│   ├── severity_model.py            # Complaint severity classification
│   ├── recurrence_model.py           # Ward recurrence risk clustering
│   └── priority_score.py            # Composite priority scoring
│
├── api/
│   └── main.py                      # API application, if implemented
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── database/                        # Database-related files
├── tests/                           # Automated tests
├── scratch/                         # Supporting scripts and experiments
├── .github/                         # GitHub configuration and workflows
├── .gitignore                       # Files excluded from Git
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

### Main Architecture Components

**1. Data Ingestion and Processing**

Municipal grievance records are loaded from the available BBMP datasets. The processing module handles data preparation, cleaning, quality checks, and the generation of structured inputs for downstream analysis.

**2. Feature Engineering**

Complaint information is converted into useful features, including severity indicators, complaint aging, and ward-level recurrence information. These features support classification and priority calculation.

**3. Severity Classification**

Text-based machine learning models use TF-IDF features to classify complaint severity. The project includes algorithms such as Naive Bayes, Logistic Regression, Decision Tree, and K-Nearest Neighbors for comparison.

**4. Ward Recurrence Risk Analysis**

K-Means clustering is used to group wards according to recurrence-related features. The resulting analysis helps identify locations that may require additional attention.

**5. Composite Priority Scoring**

The priority-scoring component combines complaint severity, recurrence risk, and complaint aging to calculate a score for ranking grievances. The score supports the creation of a prioritized dispatch worklist rather than relying only on complaint arrival order.

**6. Streamlit Dashboard**

The dashboard provides an interface for exploring grievance records, examining ward-level patterns, predicting complaint severity, viewing priority rankings, and accessing multimodal submission features where implemented.

**7. Multimodal Processing**

The planned multimodal workflow extends complaint processing to voice and image/video inputs. Speech recognition and translation can convert supported regional-language complaints into text, while YOLOv8-based object detection can identify potential civic issues from images or video frames. These outputs can then be integrated into the existing complaint analysis workflow.

**8. API and Database Integration**

The `api/` and `database/` folders provide locations for API and database-related components. FastAPI endpoints and SQLite-based storage should be described as operational components only if they have been implemented and tested. Otherwise, they remain planned extensions.

### Architecture Workflow

```text
Municipal Grievance Data
          |
          v
   Data Processing
          |
          v
   Feature Engineering
          |
          v
   Severity Classification
          |
          +--------------------+
          |                    |
          v                    v
 Ward Recurrence Analysis   Complaint Aging
          |                    |
          +---------+----------+
                    |
                    v
          Composite Priority Score
                    |
                    v
           Ranked Dispatch Queue
                    |
                    v
          Streamlit Dashboard
                    |
                    v
       Decision Support for Officials
```

The voice and vision workflows can provide additional complaint information to the processing pipeline. Their outputs should feed into the priority-scoring workflow only where that integration is implemented and verified.

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

## 📈 Project Milestone Roadmap

The project is being developed in ten stages, progressing from data preparation to a complete municipal grievance decision-support system.

### Stage 1 (10%): Dataset Ingestion and Quality Profiling

Collected and combined BBMP grievance records from 2020–2025. Examined dataset structure, missing values, complaint categories, ward distribution, and date ranges.

### Stage 2 (20%): Data Preprocessing and Cleaning

Developed data preparation workflows to handle missing values, standardize fields, remove duplicate records where appropriate, and prepare the dataset for analysis.

### Stage 3 (30%): Feature Engineering and Severity Labeling

Created relevant features from complaint information, including severity indicators, complaint aging, and attributes required for priority calculation.

### Stage 4 (40%): Machine Learning-Based Severity Classification

Developed and evaluated complaint severity classification models using TF-IDF text features and algorithms such as Naive Bayes, Logistic Regression, Decision Tree, and K-Nearest Neighbors.

### Stage 5 (50%): Ward Recurrence Analysis and Dashboard Development

Applied K-Means clustering to analyze ward-level recurrence patterns and developed the Streamlit dashboard for complaint analysis, risk visualization, prediction, and priority queue exploration.

### Stage 6 (60%): Multimodal Complaint Processing

Developed voice and vision processing workflows for civic complaints. The voice pipeline supports speech transcription, language detection, and translation, while the vision pipeline uses YOLOv8-based object detection to identify potential civic issues. The completion of these features depends on successful integration and testing.

### Stage 7 (70%): Configurable Priority Scoring

Planned the refinement of the composite priority score by allowing the weights assigned to complaint severity, ward recurrence risk, and complaint aging to be adjusted and evaluated.

### Stage 8 (80%): Baseline Comparison and Evaluation

Planned a comparative evaluation of the proposed priority-ranking system against First-In, First-Out (FIFO) processing using clearly defined evaluation metrics and experimental results.

### Stage 9 (90%): Backend API Integration

Planned the integration of a FastAPI backend with REST endpoints for complaint prediction, priority scoring, and ward recurrence risk retrieval.

### Stage 10 (100%): Database Integration and Final Validation

Planned the integration of SQLite for structured complaint storage and ingestion, followed by end-to-end testing, performance evaluation, documentation, and preparation for future municipal integration.

**Note:** The milestone percentages represent the development roadmap, not measured model accuracy or guaranteed operational readiness. Each stage should be marked complete only after its implementation and relevant testing have been verified.
