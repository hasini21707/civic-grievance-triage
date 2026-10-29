import pandas as pd
import numpy as np


def compute_historical_resolution_times(df: pd.DataFrame) -> dict:
    """
    Computes historical resolution time metrics (in days) per category and ward.
    Uses status and Grievance Date information to derive average resolution durations.
    
    Args:
        df (pd.DataFrame): Cleaned grievance DataFrame.
        
    Returns:
        dict: Mapping of (Category_Clean, Ward_Name_Clean) to average resolution days.
    """
    res_df = df.copy()
    
    # Filter resolved or closed complaints if resolution dates exist or fallback to category/ward metrics
    # In raw BBMP data, resolved complaints have status 'Closed' or 'Resolved'
    resolved = res_df[res_df["Normalized_Status"].isin(["Closed", "Resolved"])]
    
    # Category level mean pending/resolution proxy
    cat_avg = resolved.groupby("Category_Clean")["days_pending"].mean().to_dict() if not resolved.empty else {}
    global_avg = res_df["days_pending"].mean() if not res_df.empty else 7.0
    
    # Ward + Category level mean
    ward_cat_avg = resolved.groupby(["Ward_Name_Clean", "Category_Clean"])["days_pending"].mean().to_dict() if not resolved.empty else {}
    
    return {
        "global_avg": float(global_avg),
        "cat_avg": {k: float(v) for k, v in cat_avg.items()},
        "ward_cat_avg": {f"{w}_{c}": float(v) for (w, c), v in ward_cat_avg.items()}
    }


def add_resolution_time_features(df: pd.DataFrame, res_metrics: dict = None) -> pd.DataFrame:
    """
    Adds historical_avg_resolution_days and historical_resolution_score features.
    
    Args:
        df (pd.DataFrame): DataFrame to enrich.
        res_metrics (dict, optional): Historical resolution metrics dictionary.
        
    Returns:
        pd.DataFrame: DataFrame enriched with resolution time features.
    """
    enriched_df = df.copy()
    if res_metrics is None:
        res_metrics = compute_historical_resolution_times(enriched_df)
        
    global_avg = res_metrics.get("global_avg", 7.0)
    cat_avg = res_metrics.get("cat_avg", {})
    ward_cat_avg = res_metrics.get("ward_cat_avg", {})
    
    hist_days = []
    for _, row in enriched_df.iterrows():
        ward = str(row.get("Ward_Name_Clean", ""))
        cat = str(row.get("Category_Clean", ""))
        key = f"{ward}_{cat}"
        
        if key in ward_cat_avg:
            avg_d = ward_cat_avg[key]
        elif cat in cat_avg:
            avg_d = cat_avg[cat]
        else:
            avg_d = global_avg
            
        hist_days.append(round(float(avg_d), 1))
        
    enriched_df["historical_avg_resolution_days"] = hist_days
    
    # Normalize score (0 to 100) where longer historical resolution time = higher priority risk
    max_days = max(hist_days) if hist_days else 30.0
    max_days = max(max_days, 1.0)
    enriched_df["historical_resolution_score"] = (enriched_df["historical_avg_resolution_days"] / max_days * 100.0).clip(0, 100).round(1)
    
    return enriched_df
