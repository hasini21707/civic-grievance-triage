import pytest
import pandas as pd
from src.baseline_comparison import BaselineComparator


@pytest.fixture
def sample_grievances_for_baseline():
    return pd.DataFrame({
        "Complaint ID": [f"C{i}" for i in range(20)],
        "Category_Clean": ["Storm Water Drain"] * 5 + ["Roads"] * 5 + ["Electrical"] * 10,
        "Ward_Name_Clean": ["Jnanabharathi"] * 10 + ["Horamavu"] * 10,
        "severity_label": ["CRITICAL"] * 4 + ["HIGH"] * 6 + ["MEDIUM"] * 5 + ["LOW"] * 5,
        "Grievance Date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "days_pending": list(range(20, 0, -1)),
        "ward_complaint_density": [0.8] * 10 + [0.4] * 10
    })


def test_baseline_comparator(sample_grievances_for_baseline):
    comparator = BaselineComparator()
    res = comparator.compare_methods(sample_grievances_for_baseline)
    
    assert "ranked_dataframe" in res
    assert "spearman_correlations" in res
    assert "high_risk_in_top_proposed" in res
    assert "high_risk_in_top_fifo" in res
    
    ranked_df = res["ranked_dataframe"]
    assert "Rank_Proposed" in ranked_df.columns
    assert "Rank_FIFO" in ranked_df.columns
    assert "Rank_Shift_vs_FIFO" in ranked_df.columns
    
    # Verify top proposed prioritizes Critical/High complaints higher than FIFO
    assert res["high_risk_in_top_proposed"] >= res["high_risk_in_top_fifo"]
