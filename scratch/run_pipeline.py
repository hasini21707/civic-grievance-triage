import os
import sys
import time
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import load_raw_data, clean_grievance_data, generate_data_quality_report
from src.feature_engineering import extract_features
from src.severity_model import SeverityClassifier
from src.recurrence_model import WardRecurrenceAnalyzer
from src.priority_score import PriorityScorer

def main():
    print("================ PIPELINE EXECUTION STARTED ================")
    start_time = time.time()
    
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    
    # 1. Load Raw Data
    print("\n[Step 1/6] Loading raw datasets...")
    df_raw = load_raw_data(raw_dir)
    print(f"Loaded {len(df_raw):,} raw records.")
    
    # 2. Clean Data
    print("\n[Step 2/6] Cleaning and normalizing grievance data...")
    df_cleaned = clean_grievance_data(df_raw)
    print(f"Cleaned records count: {len(df_cleaned):,}")
    
    # 3. Data Quality Report
    print("\n[Step 3/6] Generating Data Quality Report...")
    report_path = os.path.join(processed_dir, "data_quality_report.json")
    report = generate_data_quality_report(df_raw, df_cleaned, report_path)
    
    # 4. Feature Engineering & Derived Severity
    print("\n[Step 4/6] Engineering features and derived severity labels...")
    df_features = extract_features(df_cleaned)
    cleaned_csv_path = os.path.join(processed_dir, "cleaned_grievances.csv")
    df_features.to_csv(cleaned_csv_path, index=False)
    print(f"Saved processed dataset to {cleaned_csv_path}")
    
    # 5. Train & Compare Machine Learning Severity Models
    print("\n[Step 5/6] Training TF-IDF + Naive Bayes & Comparison Models...")
    sample_df = df_features.sample(n=min(50000, len(df_features)), random_state=42, replace=False)
    clf = SeverityClassifier(max_features=5000)
    clf.fit_and_evaluate(sample_df["full_text"], sample_df["severity_label"])
    clf.save_model(models_dir)
    
    # 6. K-Means Ward Recurrence Analysis
    print("\n[Step 6/6] Fitting K-Means Ward Recurrence Risk Clustering...")
    rec_analyzer = WardRecurrenceAnalyzer(n_clusters=3, random_state=42)
    ward_df = rec_analyzer.fit_and_cluster(df_features)
    rec_analyzer.save_model(models_dir, processed_dir)
    
    print("\nWard Recurrence Risk Distribution:")
    print(ward_df["recurrence_level"].value_counts().to_string())
    
    # 7. Generate Daily Dispatch Worklist
    print("\n[Bonus] Generating Daily Dispatch Priority Worklist...")
    ward_map = rec_analyzer.get_ward_recurrence_map()
    scorer = PriorityScorer()
    worklist_df = scorer.generate_dispatch_worklist(df_features.sample(n=min(5000, len(df_features)), random_state=42), ward_map)
    worklist_csv_path = os.path.join(processed_dir, "dispatch_worklist.csv")
    worklist_df.to_csv(worklist_csv_path, index=False)
    print(f"Saved dispatch worklist to {worklist_csv_path}")
    
    elapsed = time.time() - start_time
    print(f"\n================ FULL PIPELINE COMPLETED IN {elapsed:.2f} SECONDS ================")

if __name__ == "__main__":
    main()
