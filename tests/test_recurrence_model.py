import pytest
import pandas as pd
import numpy as np
from src.recurrence_model import WardRecurrenceAnalyzer
from src.priority_score import PriorityScorer


@pytest.fixture
def sample_grievances():
    return pd.DataFrame({
        "Complaint ID": [f"C{i}" for i in range(100)],
        "Ward_Name_Clean": ["Ward A"] * 60 + ["Ward B"] * 30 + ["Ward C"] * 10,
        "Category_Clean": ["Electrical"] * 50 + ["Roads"] * 50,
        "Sub_Category_Clean": ["Streetlight"] * 40 + ["Pothole"] * 60
    })


def test_ward_recurrence_analyzer(sample_grievances):
    analyzer = WardRecurrenceAnalyzer(n_clusters=3, random_state=42)
    ward_df = analyzer.fit_and_cluster(sample_grievances)
    
    assert len(ward_df) == 3
    assert "recurrence_score" in ward_df.columns
    assert "recurrence_level" in ward_df.columns
    
    # Recurrence scores should be within range 0.0 to 1.0
    assert ward_df["recurrence_score"].min() >= 0.0
    assert ward_df["recurrence_score"].max() <= 1.0
    
    # Ward A has 60 complaints, should have higher recurrence score than Ward C (10 complaints)
    ward_a_score = ward_df[ward_df["Ward_Name_Clean"] == "Ward A"]["recurrence_score"].values[0]
    ward_c_score = ward_df[ward_df["Ward_Name_Clean"] == "Ward C"]["recurrence_score"].values[0]
    assert ward_a_score >= ward_c_score


def test_priority_scorer_bounds():
    scorer = PriorityScorer(weight_severity=0.50, weight_recurrence=0.30, weight_aging=0.20)
    
    # Critical test case
    res_crit = scorer.calculate_priority_score("CRITICAL", 1.0, 30.0)
    assert res_crit["priority_score"] == 100.0
    assert res_crit["priority_level"] == "CRITICAL"
    
    # Low test case
    res_low = scorer.calculate_priority_score("LOW", 0.0, 0.0)
    assert res_low["priority_score"] == 12.5  # (25*0.5 + 0 + 0)
    assert res_low["priority_level"] == "LOW"
    
    # Verify bounds [0, 100]
    assert 0.0 <= res_crit["priority_score"] <= 100.0
    assert 0.0 <= res_low["priority_score"] <= 100.0
