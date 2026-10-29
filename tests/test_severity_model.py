import pytest
import pandas as pd
from src.feature_engineering import assign_derived_severity, extract_features
from src.severity_model import SeverityClassifier


def test_assign_derived_severity():
    row_critical = {"Category": "Storm Water Drain(SWD)", "Sub Category": "Open Manhole", "full_text": "open manhole dangerous"}
    row_high = {"Category": "Road Maintenance", "Sub Category": "Potholes", "full_text": "major pothole severe road blockage"}
    row_low = {"Category": "Health Dept", "Sub Category": "Sweeping not done", "full_text": "sweeping not done minor"}
    
    assert assign_derived_severity(row_critical) == "CRITICAL"
    assert assign_derived_severity(row_high) == "HIGH"
    assert assign_derived_severity(row_low) == "LOW"


def test_extract_features():
    df = pd.DataFrame({
        "Complaint ID": ["C1", "C2"],
        "Category": ["Electrical", "Storm Water Drain"],
        "Sub Category": ["Street Light Not Working", "Open Manhole"],
        "Category_Clean": ["Electrical", "Storm Water Drain"],
        "Sub_Category_Clean": ["Street Light Not Working", "Open Manhole"],
        "Ward_Name_Clean": ["Jnanabharathi", "Jnanabharathi"],
        "Grievance Date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "full_text": ["street light not working routine", "open manhole dangerous overflow"]
    })
    
    feats = extract_features(df)
    
    assert "severity_label" in feats.columns
    assert "days_pending" in feats.columns
    assert "ward_complaint_count" in feats.columns
    assert "ward_complaint_density" in feats.columns
    assert feats["severity_label"].iloc[1] == "CRITICAL"
    assert feats["ward_complaint_count"].iloc[0] == 2


def test_severity_classifier_pipeline(tmp_path):
    texts = pd.Series([
        "open manhole dangerous electrocution fire",
        "major pothole severe road blockage",
        "street light not working routine maintenance",
        "sweeping not done minor trash"
    ] * 10)
    
    labels = pd.Series(["CRITICAL", "HIGH", "MEDIUM", "LOW"] * 10)
    
    clf = SeverityClassifier(max_features=100)
    metrics = clf.fit_and_evaluate(texts, labels, test_size=0.2, random_state=42)
    
    assert "Naive Bayes" in metrics
    assert "accuracy" in metrics["Naive Bayes"]
    assert metrics["Naive Bayes"]["accuracy"] >= 0.0
    
    preds = clf.predict_severity(["open manhole fire"])
    assert len(preds) == 1
    assert preds[0] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    
    # Save and Load test
    save_dir = str(tmp_path / "models")
    clf.save_model(save_dir)
    
    clf_new = SeverityClassifier()
    clf_new.load_model(save_dir)
    preds_loaded = clf_new.predict_severity(["open manhole fire"])
    assert preds_loaded == preds
