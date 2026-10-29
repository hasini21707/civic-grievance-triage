import os
import nbformat as nbf

nb_dir = r"c:\Users\hasin\OneDrive\DSP civic grievance project\notebooks"
os.makedirs(nb_dir, exist_ok=True)

# 1. 01_EDA.ipynb
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("# 01. Exploratory Data Analysis (EDA)\n## Civic Grievance Triage & Resolution-Priority Scoring System\nThis notebook analyzes raw and processed BBMP civic grievance dataset (2020-2025)."),
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

df = pd.read_csv('../data/processed/cleaned_grievances.csv', low_memory=False)
with open('../data/processed/data_quality_report.json', 'r') as f:
    report = json.load(f)

print(f"Total Cleaned Records: {len(df):,}")
print(f"Total Wards: {report['number_of_wards']}")
print(f"Total Categories: {report['number_of_categories']}")
"""),
    nbf.v4.new_markdown_cell("### Category Distribution"),
    nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
df['Category_Clean'].value_counts().head(10).plot(kind='barh', color='teal')
plt.title('Top 10 Grievance Categories in BBMP')
plt.xlabel('Complaint Count')
plt.ylabel('Category')
plt.gca().invert_yaxis()
plt.show()
"""),
    nbf.v4.new_markdown_cell("### Ward Distribution"),
    nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
df['Ward_Name_Clean'].value_counts().head(10).plot(kind='bar', color='coral')
plt.title('Top 10 Wards by Grievance Volume')
plt.xlabel('Ward Name')
plt.ylabel('Complaint Count')
plt.xticks(rotation=45)
plt.show()
"""),
    nbf.v4.new_markdown_cell("### Grievance Status Breakdown"),
    nbf.v4.new_code_cell("""print(df['Normalized_Status'].value_counts())
plt.figure(figsize=(6, 6))
df['Normalized_Status'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#4CAF50', '#FF9800', '#F44336', '#2196F3'])
plt.title('Grievance Status Distribution')
plt.ylabel('')
plt.show()
""")
]

with open(os.path.join(nb_dir, "01_EDA.ipynb"), "w", encoding="utf-8") as f:
    nbf.write(nb1, f)
print("Generated notebooks/01_EDA.ipynb")

# 2. 02_Preprocessing.ipynb
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("# 02. Data Cleaning & Preprocessing Pipeline\nDemonstrating deduplication, null handling, date parsing, and derived severity label creation."),
    nbf.v4.new_code_cell("""import sys
import os
sys.path.append('..')

from src.data_processing import load_raw_data, clean_grievance_data, generate_data_quality_report
from src.feature_engineering import extract_features

raw_dir = '../data/raw'
df_raw = load_raw_data(raw_dir)
df_clean = clean_grievance_data(df_raw)
df_feat = extract_features(df_clean)

print("Preprocessing complete!")
print(f"Raw records: {len(df_raw):,}")
print(f"Clean records: {len(df_feat):,}")
print("Derived Severity Distribution:")
print(df_feat['severity_label'].value_counts())
""")
]

with open(os.path.join(nb_dir, "02_Preprocessing.ipynb"), "w", encoding="utf-8") as f:
    nbf.write(nb2, f)
print("Generated notebooks/02_Preprocessing.ipynb")

# 3. 03_Severity_Classification.ipynb
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("# 03. Machine Learning Severity Classification\nTraining TF-IDF + Naive Bayes (Primary) and comparing against Logistic Regression, Decision Tree, and KNN."),
    nbf.v4.new_code_cell("""import sys
import os
import pandas as pd
import joblib
sys.path.append('..')

from src.severity_model import SeverityClassifier

# Load metrics report from trained models
metrics_path = '../models/severity_metrics_report.joblib'
metrics = joblib.load(metrics_path)

metrics_df = pd.DataFrame(metrics).T[['accuracy', 'precision', 'recall', 'f1_score']]
print("=== MODEL COMPARISON TABLE ===")
print(metrics_df)
"""),
    nbf.v4.new_markdown_cell("### Test Sample Predictions with Primary Naive Bayes Model"),
    nbf.v4.new_code_cell("""clf = SeverityClassifier()
clf.load_model('../models')

test_samples = [
    "open manhole dangerous exposed wire live cable",
    "major pothole severe road blockage near junction",
    "street light not working since 3 days",
    "sweeping not done minor garbage near corner"
]

preds = clf.predict_severity(test_samples)
probs = clf.predict_proba(test_samples)

for text, pred, prob in zip(test_samples, preds, probs):
    print(f"Text: '{text}' -> Predicted Severity: [{pred}]")
    print(f"      Probabilities: {prob}\n")
""")
]

with open(os.path.join(nb_dir, "03_Severity_Classification.ipynb"), "w", encoding="utf-8") as f:
    nbf.write(nb3, f)
print("Generated notebooks/03_Severity_Classification.ipynb")
