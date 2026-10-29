import os
import glob
import json
import re
import pandas as pd
import numpy as np

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    # Pre-download NLTK resources safely if needed
    try:
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
    STEMMER = PorterStemmer()
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    STOPWORDS = set()
    STEMMER = None


def preprocess_text_nltk(text: str) -> str:
    """
    Preprocesses raw grievance text using NLTK:
    - Lowercasing & non-alphanumeric removal
    - Stopword removal
    - Porter Stemming
    
    Args:
        text (str): Raw input text string.
        
    Returns:
        str: Preprocessed cleaned string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = re.findall(r"\b[a-z]{2,}\b", text)
    if NLTK_AVAILABLE and STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS]
        if STEMMER:
            tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
    Loads all raw BBMP grievance CSV files (2020-2025) from data_dir.
    
    Args:
        data_dir (str): Path to directory containing raw CSV files.
        
    Returns:
        pd.DataFrame: Combined raw DataFrame.
    """
    csv_pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(csv_pattern))
    
    if not files:
        raise FileNotFoundError(f"No CSV dataset files found in {data_dir}")
        
    dfs = []
    for filepath in files:
        sub_df = pd.read_csv(filepath, low_memory=False)
        dfs.append(sub_df)
        
    raw_df = pd.concat(dfs, ignore_index=True)
    return raw_df


def clean_grievance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw civic grievance DataFrame.
    - Removes duplicate records based on Complaint ID.
    - Parses Grievance Date to datetime.
    - Handles missing values across columns.
    - Normalizes strings (lowercasing, whitespace trimming).
    - Constructs full_text field combining Category, Sub Category, and Staff Remarks.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # 1. Deduplication
    if "Complaint ID" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=["Complaint ID"])
        
    # 2. Date conversion
    if "Grievance Date" in cleaned_df.columns:
        cleaned_df["Grievance Date"] = pd.to_datetime(cleaned_df["Grievance Date"], errors="coerce")
        # Remove records with completely invalid dates if any exist
        cleaned_df = cleaned_df.dropna(subset=["Grievance Date"])
        
    # 3. Missing Value Handling
    cleaned_df["Grievance Status"] = cleaned_df["Grievance Status"].fillna("Registered").astype(str).str.strip()
    cleaned_df["Staff Remarks"] = cleaned_df["Staff Remarks"].fillna("No remarks").astype(str).str.strip()
    cleaned_df["Staff Name"] = cleaned_df["Staff Name"].fillna("Unassigned").astype(str).str.strip()
    cleaned_df["Category"] = cleaned_df["Category"].fillna("Others").astype(str).str.strip()
    cleaned_df["Sub Category"] = cleaned_df["Sub Category"].fillna("General").astype(str).str.strip()
    cleaned_df["Ward Name"] = cleaned_df["Ward Name"].fillna("Unknown Ward").astype(str).str.strip()

    # 4. Text Normalization
    cleaned_df["Category_Clean"] = cleaned_df["Category"].str.strip().str.title()
    cleaned_df["Sub_Category_Clean"] = cleaned_df["Sub Category"].str.strip()
    cleaned_df["Ward_Name_Clean"] = cleaned_df["Ward Name"].str.strip().str.title()
    
    # 5. Status Normalization
    status_map = {
        "Closed": "Closed",
        "Registered": "Registered",
        "Rejected": "Rejected",
        "Non Relevant": "Rejected",
        "Resolved": "Resolved",
        "ReOpen": "In Progress",
        "In Progress": "In Progress",
        "Long Term Solution": "In Progress"
    }
    cleaned_df["Normalized_Status"] = cleaned_df["Grievance Status"].map(lambda x: status_map.get(str(x).strip(), "Registered"))
    
    # 6. Combined Search & Feature Text
    cleaned_df["full_text"] = (
        cleaned_df["Category_Clean"] + " " + 
        cleaned_df["Sub_Category_Clean"] + " " + 
        cleaned_df["Staff Remarks"]
    ).str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True)
    
    return cleaned_df


def generate_data_quality_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, output_path: str = None) -> dict:
    """
    Generates data quality statistics comparing raw and cleaned datasets.
    
    Args:
        raw_df (pd.DataFrame): Input raw DataFrame.
        cleaned_df (pd.DataFrame): Output cleaned DataFrame.
        output_path (str, optional): Path to save JSON report.
        
    Returns:
        dict: Data quality metric dictionary.
    """
    raw_total = len(raw_df)
    cleaned_total = len(cleaned_df)
    duplicates_removed = raw_total - raw_df["Complaint ID"].nunique() if "Complaint ID" in raw_df.columns else 0
    
    missing_raw = raw_df.isnull().sum().to_dict()
    missing_cleaned = cleaned_df.isnull().sum().to_dict()
    
    report = {
        "number_of_original_records": raw_total,
        "number_of_cleaned_records": cleaned_total,
        "number_of_duplicates": duplicates_removed,
        "number_of_wards": int(cleaned_df["Ward_Name_Clean"].nunique()),
        "number_of_categories": int(cleaned_df["Category_Clean"].nunique()),
        "number_of_subcategories": int(cleaned_df["Sub_Category_Clean"].nunique()),
        "date_range": {
            "min_date": str(cleaned_df["Grievance Date"].min()),
            "max_date": str(cleaned_df["Grievance Date"].max())
        },
        "missing_values_raw": {k: int(v) for k, v in missing_raw.items()},
        "missing_values_cleaned": {k: int(v) for k, v in missing_cleaned.items()},
        "status_distribution": {k: int(v) for k, v in cleaned_df["Normalized_Status"].value_counts().to_dict().items()}
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
    return report
