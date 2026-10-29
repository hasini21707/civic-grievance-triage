import os
import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_raw_data, clean_grievance_data, generate_data_quality_report


@pytest.fixture
def sample_raw_data():
    data = {
        "Complaint ID": ["C101", "C102", "C103", "C101"],  # C101 is duplicate
        "Category": ["Electrical", "Solid Waste (Garbage) Related", "Road Maintenance(Engg)", "Electrical"],
        "Sub Category": ["Street Light Not Working", "Garbage dump", "Potholes", "Street Light Not Working"],
        "Grievance Date": ["2023-05-10 10:00:00", "2023-05-11 14:30:00", "2023-05-12 09:15:00", "2023-05-10 10:00:00"],
        "Ward Name": ["Jnanabharathi Ward", "Horamavu", "Begur", "Jnanabharathi Ward"],
        "Grievance Status": ["Closed", "Registered", None, "Closed"],
        "Staff Remarks": ["Fixed bulb", None, "Pothole filled", "Fixed bulb"],
        "Staff Name": ["John Doe", "Jane Smith", None, "John Doe"]
    }
    return pd.DataFrame(data)


def test_clean_grievance_data(sample_raw_data):
    cleaned = clean_grievance_data(sample_raw_data)
    
    # Check deduplication
    assert len(cleaned) == 3
    assert set(cleaned["Complaint ID"]) == {"C101", "C102", "C103"}
    
    # Check missing value handling
    assert cleaned["Staff Remarks"].isnull().sum() == 0
    assert cleaned["Staff Name"].isnull().sum() == 0
    assert cleaned["Grievance Status"].isnull().sum() == 0
    
    # Check datetime parsing
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Grievance Date"])
    
    # Check full_text feature creation
    assert "full_text" in cleaned.columns
    assert isinstance(cleaned["full_text"].iloc[0], str)


def test_generate_data_quality_report(sample_raw_data):
    cleaned = clean_grievance_data(sample_raw_data)
    report = generate_data_quality_report(sample_raw_data, cleaned)
    
    assert report["number_of_original_records"] == 4
    assert report["number_of_cleaned_records"] == 3
    assert report["number_of_duplicates"] == 1
    assert report["number_of_wards"] == 3
    assert report["number_of_categories"] == 3
