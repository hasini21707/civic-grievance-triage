import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class WardRecurrenceAnalyzer:
    """
    Ward Recurrence Risk Analyzer using K-Means Clustering.
    Groups municipal wards into Low, Medium, and High Recurrence Risk levels
    based on historical complaint frequency, density, and category diversity.
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.elbow_scores = {}
        self.silhouette_scores = {}
        self.ward_summary = pd.DataFrame()
        self.is_fitted = False

    def compute_ward_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates ward-level historical metrics:
        - complaint_count: Total complaints in the ward
        - category_diversity: Number of unique categories reported
        - top_category_ratio: Ratio of the single largest category in the ward
        - repeat_subcategory_count: Count of repeat subcategory occurrences
        
        Args:
            df (pd.DataFrame): Cleaned grievance DataFrame.
            
        Returns:
            pd.DataFrame: Ward feature matrix (one row per ward).
        """
        ward_grp = df.groupby("Ward_Name_Clean")
        
        ward_metrics = []
        for ward_name, group in ward_grp:
            total_count = len(group)
            cat_count = group["Category_Clean"].nunique()
            top_cat_freq = group["Category_Clean"].value_counts().iloc[0] if total_count > 0 else 0
            top_cat_ratio = top_cat_freq / total_count if total_count > 0 else 0
            subcat_repeats = (group["Sub_Category_Clean"].value_counts() > 1).sum()
            
            ward_metrics.append({
                "Ward_Name_Clean": ward_name,
                "complaint_count": total_count,
                "category_diversity": cat_count,
                "top_category_ratio": round(top_cat_ratio, 4),
                "repeat_subcategory_count": subcat_repeats
            })
            
        ward_df = pd.DataFrame(ward_metrics)
        max_count = ward_df["complaint_count"].max() if not ward_df.empty else 1
        ward_df["complaint_density"] = (ward_df["complaint_count"] / max_count).round(4)
        
        return ward_df

    def fit_and_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits K-Means model on ward feature matrix and calculates Elbow & Silhouette metrics.
        
        Args:
            df (pd.DataFrame): Cleaned grievance DataFrame.
            
        Returns:
            pd.DataFrame: Ward feature DataFrame with assigned recurrence scores and levels.
        """
        ward_df = self.compute_ward_features(df)
        
        feature_cols = ["complaint_count", "complaint_density", "category_diversity", "repeat_subcategory_count"]
        X = ward_df[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute Elbow & Silhouette for k=2..6
        self.elbow_scores = {}
        self.silhouette_scores = {}
        for k in range(2, min(7, len(ward_df))):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = km.fit_predict(X_scaled)
            self.elbow_scores[k] = float(km.inertia_)
            self.silhouette_scores[k] = float(silhouette_score(X_scaled, cluster_labels))
            
        # Fit primary n_clusters model
        ward_df["cluster"] = self.model.fit_predict(X_scaled)
        
        # Order clusters by mean complaint count so cluster ranking is deterministic (0 = Low, 1 = Med, 2 = High)
        cluster_means = ward_df.groupby("cluster")["complaint_count"].mean().sort_values().index
        cluster_map = {old_c: new_rank for new_rank, old_c in enumerate(cluster_means)}
        ward_df["cluster_rank"] = ward_df["cluster"].map(cluster_map)
        
        # Calculate normalized recurrence score (0.0 to 1.0)
        min_rank = ward_df["cluster_rank"].min()
        max_rank = ward_df["cluster_rank"].max()
        if max_rank > min_rank:
            ward_df["recurrence_score"] = ((ward_df["cluster_rank"] - min_rank) / (max_rank - min_rank)).round(2)
        else:
            ward_df["recurrence_score"] = 0.5
            
        # Assign text risk levels
        level_map = {0: "Low Recurrence", 1: "Medium Recurrence", 2: "High Recurrence"}
        ward_df["recurrence_level"] = ward_df["cluster_rank"].map(lambda r: level_map.get(r, "Medium Recurrence"))
        
        self.ward_summary = ward_df
        self.is_fitted = True
        return ward_df

    def get_ward_recurrence_map(self):
        """Returns dict mapping Ward_Name_Clean to its recurrence_score and recurrence_level."""
        if not self.is_fitted:
            raise RuntimeError("Recurrence model is not fitted yet.")
        
        res = {}
        for _, row in self.ward_summary.iterrows():
            res[row["Ward_Name_Clean"]] = {
                "recurrence_score": float(row["recurrence_score"]),
                "recurrence_level": str(row["recurrence_level"]),
                "complaint_count": int(row["complaint_count"])
            }
        return res

    def save_model(self, models_dir: str, processed_dir: str):
        """Saves model artifacts and exported ward recurrence CSV."""
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(models_dir, "kmeans_recurrence_model.joblib"))
        joblib.dump(self.scaler, os.path.join(models_dir, "recurrence_scaler.joblib"))
        joblib.dump({
            "elbow": self.elbow_scores,
            "silhouette": self.silhouette_scores,
            "n_clusters": self.n_clusters
        }, os.path.join(models_dir, "recurrence_metrics.joblib"))
        
        self.ward_summary.to_csv(os.path.join(processed_dir, "ward_recurrence.csv"), index=False)
        print(f"Saved ward recurrence model to {models_dir} and ward_recurrence.csv to {processed_dir}")
