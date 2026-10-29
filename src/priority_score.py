import pandas as pd
import numpy as np


class PriorityScorer:
    """
    Composite Priority Scoring System for Civic Grievance Triage.
    Combines Severity (50%), Recurrence Risk (30%), and Aging (20%) into a 0-100 score.
    """
    
    SEVERITY_NUMERIC_MAP = {
        "CRITICAL": 100.0,
        "HIGH": 75.0,
        "MEDIUM": 50.0,
        "LOW": 25.0
    }

    def __init__(self, weight_severity: float = 0.50, weight_recurrence: float = 0.30, weight_aging: float = 0.20, max_aging_days: float = 30.0):
        if not np.isclose(weight_severity + weight_recurrence + weight_aging, 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.w_severity = weight_severity
        self.w_recurrence = weight_recurrence
        self.w_aging = weight_aging
        self.max_aging_days = max_aging_days

    def calculate_priority_score(self, severity_str: str, recurrence_score: float, days_pending: float) -> dict:
        """
        Calculates composite priority score (0-100) and priority level for a single grievance.
        
        Args:
            severity_str (str): 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'.
            recurrence_score (float): Normalized recurrence risk score (0.0 to 1.0).
            days_pending (float): Aging in days.
            
        Returns:
            dict: Detailed score components, composite priority score, level, and natural explanation.
        """
        # 1. Severity Contribution (0-100)
        sev_norm = float(self.SEVERITY_NUMERIC_MAP.get(str(severity_str).upper(), 50.0))
        
        # 2. Recurrence Contribution (0-100)
        rec_norm = float(np.clip(recurrence_score * 100.0, 0.0, 100.0))
        
        # 3. Aging Contribution (0-100)
        aging_norm = float(np.clip((days_pending / self.max_aging_days) * 100.0, 0.0, 100.0))
        
        # 4. Composite Priority Score
        priority_score = float(np.clip(
            (sev_norm * self.w_severity) + 
            (rec_norm * self.w_recurrence) + 
            (aging_norm * self.w_aging),
            0.0, 100.0
        ))
        priority_score = round(priority_score, 1)
        
        # 5. Priority Level Mapping
        if priority_score >= 80.0:
            priority_level = "CRITICAL"
        elif priority_score >= 60.0:
            priority_level = "HIGH"
        elif priority_score >= 40.0:
            priority_level = "MEDIUM"
        else:
            priority_level = "LOW"
            
        # 6. Natural Language Explanation
        explanation = (
            f"Complaint received '{priority_level}' priority (Score: {priority_score}/100) "
            f"due to {severity_str} severity contribution ({sev_norm * self.w_severity:.1f} pts), "
            f"ward recurrence risk contribution ({rec_norm * self.w_recurrence:.1f} pts), "
            f"and {days_pending:.1f} days pending aging contribution ({aging_norm * self.w_aging:.1f} pts)."
        )
        
        return {
            "severity_norm": sev_norm,
            "recurrence_norm": rec_norm,
            "aging_norm": aging_norm,
            "priority_score": priority_score,
            "priority_level": priority_level,
            "explanation": explanation
        }

    def generate_dispatch_worklist(self, df: pd.DataFrame, ward_recurrence_map: dict = None) -> pd.DataFrame:
        """
        Generates daily dispatch-ranked worklist sorted descending by Priority Score.
        
        Args:
            df (pd.DataFrame): DataFrame with severity_label, days_pending, Ward_Name_Clean, Sub_Category_Clean.
            ward_recurrence_map (dict, optional): Map of ward names to recurrence scores.
            
        Returns:
            pd.DataFrame: Ranked dispatch DataFrame.
        """
        work_df = df.copy()
        
        priority_scores = []
        priority_levels = []
        rec_scores = []
        explanations = []
        
        for _, row in work_df.iterrows():
            sev = row.get("severity_label", "MEDIUM")
            ward_name = row.get("Ward_Name_Clean", "")
            
            if ward_recurrence_map and ward_name in ward_recurrence_map:
                rec_score = ward_recurrence_map[ward_name]["recurrence_score"]
            else:
                rec_score = float(row.get("ward_complaint_density", 0.5))
                
            days_p = float(row.get("days_pending", 0.0))
            
            res = self.calculate_priority_score(sev, rec_score, days_p)
            
            priority_scores.append(res["priority_score"])
            priority_levels.append(res["priority_level"])
            rec_scores.append(round(rec_score * 100.0, 1))
            explanations.append(res["explanation"])
            
        work_df["recurrence_score"] = rec_scores
        work_df["priority_score"] = priority_scores
        work_df["priority_level"] = priority_levels
        work_df["explanation"] = explanations
        
        # Sort descending by priority_score, then by days_pending
        work_df = work_df.sort_values(by=["priority_score", "days_pending"], ascending=[False, False]).reset_index(drop=True)
        work_df.insert(0, "Rank", work_df.index + 1)
        
        return work_df
