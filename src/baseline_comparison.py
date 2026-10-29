import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.priority_score import PriorityScorer


class BaselineComparator:
    """
    Evaluates proposed composite priority triage system against baseline methodologies:
    1. FIFO (First-In, First-Out)
    2. Severity-Only Priority
    3. Severity + Aging Priority
    4. Proposed Composite (Severity + Recurrence + Aging)
    """
    
    SEVERITY_WEIGHTS = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

    def __init__(self, priority_scorer: PriorityScorer = None):
        self.scorer = priority_scorer or PriorityScorer()

    def compare_methods(self, df: pd.DataFrame, ward_recurrence_map: dict = None) -> dict:
        """
        Runs 4 ranking methods on dataset and returns rank shift evaluation.
        
        Args:
            df (pd.DataFrame): Grievance DataFrame.
            ward_recurrence_map (dict, optional): Ward recurrence scores mapping.
            
        Returns:
            dict: Rank tables, Spearman correlations, and high-risk prioritization gains.
        """
        sample_df = df.copy()
        
        # 1. Proposed Composite Ranking
        proposed_df = self.scorer.generate_dispatch_worklist(sample_df, ward_recurrence_map)
        proposed_df["Rank_Proposed"] = proposed_df["Rank"]
        
        # 2. FIFO Ranking (Oldest Grievance Date First)
        fifo_df = sample_df.sort_values(by="Grievance Date", ascending=True).reset_index(drop=True)
        fifo_df["Rank_FIFO"] = fifo_df.index + 1
        
        # 3. Severity-Only Ranking
        sev_order = sample_df.copy()
        sev_order["sev_num"] = sev_order["severity_label"].map(self.SEVERITY_WEIGHTS).fillna(2)
        sev_df = sev_order.sort_values(by="sev_num", ascending=False).reset_index(drop=True)
        sev_df["Rank_Severity_Only"] = sev_df.index + 1
        
        # 4. Severity + Aging Ranking (70% Severity + 30% Aging)
        sev_aging_order = sample_df.copy()
        sev_aging_order["sev_norm"] = sev_aging_order["severity_label"].map(PriorityScorer.SEVERITY_NUMERIC_MAP).fillna(50)
        sev_aging_order["aging_norm"] = (sev_aging_order["days_pending"] / 30.0 * 100.0).clip(0, 100)
        sev_aging_order["score_sev_aging"] = (sev_aging_order["sev_norm"] * 0.70) + (sev_aging_order["aging_norm"] * 0.30)
        sev_aging_df = sev_aging_order.sort_values(by="score_sev_aging", ascending=False).reset_index(drop=True)
        sev_aging_df["Rank_Severity_Aging"] = sev_aging_df.index + 1

        # Merge Ranks for Correlation and Rank Shift Analysis
        merged = proposed_df[["Complaint ID", "Category_Clean", "Ward_Name_Clean", "severity_label", "priority_score", "Rank_Proposed"]].copy()
        merged = merged.merge(fifo_df[["Complaint ID", "Rank_FIFO"]], on="Complaint ID", how="left")
        merged = merged.merge(sev_df[["Complaint ID", "Rank_Severity_Only"]], on="Complaint ID", how="left")
        merged = merged.merge(sev_aging_df[["Complaint ID", "Rank_Severity_Aging"]], on="Complaint ID", how="left")

        # Rank Shift Metrics
        merged["Rank_Shift_vs_FIFO"] = merged["Rank_FIFO"] - merged["Rank_Proposed"]
        
        # Critical & High Complaints prioritized in Top 10%
        top_10_pct_count = max(1, int(len(merged) * 0.10))
        top_proposed = merged.head(top_10_pct_count)
        top_fifo = merged.sort_values(by="Rank_FIFO").head(top_10_pct_count)
        
        proposed_critical_high = len(top_proposed[top_proposed["severity_label"].isin(["CRITICAL", "HIGH"])])
        fifo_critical_high = len(top_fifo[top_fifo["severity_label"].isin(["CRITICAL", "HIGH"])])
        
        # Spearman Correlations
        corr_fifo, _ = spearmanr(merged["Rank_Proposed"], merged["Rank_FIFO"])
        corr_sev, _ = spearmanr(merged["Rank_Proposed"], merged["Rank_Severity_Only"])
        corr_sev_aging, _ = spearmanr(merged["Rank_Proposed"], merged["Rank_Severity_Aging"])

        return {
            "ranked_dataframe": merged,
            "top_10_pct_count": top_10_pct_count,
            "high_risk_in_top_proposed": proposed_critical_high,
            "high_risk_in_top_fifo": fifo_critical_high,
            "high_risk_gain_ratio": round(proposed_critical_high / max(1, fifo_critical_high), 2),
            "avg_rank_shift_critical": float(merged[merged["severity_label"] == "CRITICAL"]["Rank_Shift_vs_FIFO"].mean() if not merged[merged["severity_label"] == "CRITICAL"].empty else 0.0),
            "spearman_correlations": {
                "Proposed_vs_FIFO": round(float(corr_fifo), 4),
                "Proposed_vs_Severity_Only": round(float(corr_sev), 4),
                "Proposed_vs_Severity_Aging": round(float(corr_sev_aging), 4)
            }
        }
