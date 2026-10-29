import pandas as pd
import numpy as np
import datetime

# Configurable keyword lists for severity estimation
CRITICAL_KEYWORDS = [
    "open manhole", "manhole", "electrocution", "fire", "collapse", "accident",
    "dangerous", "exposed wire", "major leakage", "overflow", "burst pipe",
    "sewage overflow", "hazard", "electric pole", "downed power lines", "power lines",
    "power line", "snapped wire", "snapped electric", "live wire", "broken wire",
    "broken cable", "hanging wire", "wire fallen", "sparking", "current shock"
]

HIGH_KEYWORDS = [
    "severe", "major pothole", "blocked drain", "large leakage", "dangerous road",
    "fallen tree", "dead tree", "tree collapse", "water contamination", "drain blockage",
    "flooding", "road cave"
]

MEDIUM_KEYWORDS = [
    "pothole", "garbage", "dump", "streetlight", "street light", "drainage", "leakage",
    "branches", "debris", "garbage vehicle", "stagnant water", "dog menace"
]

LOW_KEYWORDS = [
    "minor", "small", "maintenance", "sweeping", "banner", "hoarding", "cleaning",
    "name board", "trimming"
]

CRITICAL_SUBCATEGORIES = [
    "storm water drain", "swd", "open manhole", "electrocution", "building collapse",
    "sewage overflow", "exposed cable", "fire emergency"
]

HIGH_SUBCATEGORIES = [
    "potholes", "blocked drain", "removal of dead/fallen trees", "road cave in",
    "major leakage", "water supply interruption"
]

LOW_SUBCATEGORIES = [
    "sweeping not done", "removal of illegal banners", "routine maintenance"
]


def assign_derived_severity(row: pd.Series) -> str:
    """
    Creates derived severity labels (CRITICAL, HIGH, MEDIUM, LOW) using category,
    subcategory, text keywords, and structured risk indicators.
    
    Args:
        row (pd.Series): DataFrame row containing Category, Sub Category, and full_text.
        
    Returns:
        str: Severity label ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
    """
    text = str(row.get("full_text", "")).lower()
    sub_cat = str(row.get("Sub Category", "")).lower()
    cat = str(row.get("Category", "")).lower()
    
    # 1. Critical Check
    for kw in CRITICAL_KEYWORDS:
        if kw in text:
            return "CRITICAL"
    for sc in CRITICAL_SUBCATEGORIES:
        if sc in sub_cat or sc in cat:
            return "CRITICAL"
            
    # 2. High Check
    for kw in HIGH_KEYWORDS:
        if kw in text:
            return "HIGH"
    for sc in HIGH_SUBCATEGORIES:
        if sc in sub_cat:
            return "HIGH"
            
    # 3. Low Check
    for sc in LOW_SUBCATEGORIES:
        if sc in sub_cat:
            return "LOW"
    for kw in LOW_KEYWORDS:
        if kw in text and not any(m in text for m in MEDIUM_KEYWORDS):
            return "LOW"
            
    # 4. Default to Medium for standard municipal complaints (Garbage, Streetlights, etc.)
    return "MEDIUM"


def extract_features(df: pd.DataFrame, reference_date: datetime.datetime = None) -> pd.DataFrame:
    """
    Generates derived features required for scoring and classification:
    - Derived Severity Label
    - Days Pending (Aging)
    - Ward Complaint Count
    - Ward Complaint Density (Normalized)
    - Category Frequency
    - Text Keyword Presence Flags
    - Text Length
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        reference_date (datetime.datetime, optional): Base date for days_pending.
        
    Returns:
        pd.DataFrame: DataFrame enriched with derived features.
    """
    feat_df = df.copy()
    
    # 1. Derived Severity Labeling
    if "severity_label" not in feat_df.columns:
        feat_df["severity_label"] = feat_df.apply(assign_derived_severity, axis=1)
        
    # 2. Days Pending (Aging)
    if reference_date is None:
        if "Grievance Date" in feat_df.columns and not feat_df["Grievance Date"].empty:
            reference_date = feat_df["Grievance Date"].max() + pd.Timedelta(days=1)
        else:
            reference_date = pd.Timestamp.now()
            
    if "Grievance Date" in feat_df.columns:
        feat_df["days_pending"] = (reference_date - feat_df["Grievance Date"]).dt.total_seconds() / (24 * 3600)
        feat_df["days_pending"] = feat_df["days_pending"].clip(lower=0).round(1)
    else:
        feat_df["days_pending"] = 0.0

    # 3. Ward Historical Complaint Count & Density
    ward_counts = feat_df["Ward_Name_Clean"].value_counts().to_dict()
    feat_df["ward_complaint_count"] = feat_df["Ward_Name_Clean"].map(ward_counts).fillna(1)
    max_ward_count = max(ward_counts.values()) if ward_counts else 1
    feat_df["ward_complaint_density"] = (feat_df["ward_complaint_count"] / max_ward_count).round(4)

    # 4. Category Frequency
    cat_counts = feat_df["Category_Clean"].value_counts().to_dict()
    feat_df["category_frequency"] = feat_df["Category_Clean"].map(cat_counts).fillna(1)

    # 5. Keyword Binary Flags
    feat_df["has_critical_kw"] = feat_df["full_text"].apply(lambda t: int(any(kw in str(t) for kw in CRITICAL_KEYWORDS)))
    feat_df["has_high_kw"] = feat_df["full_text"].apply(lambda t: int(any(kw in str(t) for kw in HIGH_KEYWORDS)))
    feat_df["has_medium_kw"] = feat_df["full_text"].apply(lambda t: int(any(kw in str(t) for kw in MEDIUM_KEYWORDS)))
    feat_df["has_low_kw"] = feat_df["full_text"].apply(lambda t: int(any(kw in str(t) for kw in LOW_KEYWORDS)))
    
    # 6. Text Length
    feat_df["text_length"] = feat_df["full_text"].str.len().fillna(0)
    
    return feat_df
