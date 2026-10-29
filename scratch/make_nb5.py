import os
import nbformat as nbf

nb_dir = r"c:\Users\hasin\OneDrive\DSP civic grievance project\notebooks"
nb5 = nbf.v4.new_notebook()
nb5.cells = [
    nbf.v4.new_markdown_cell("# 05. Baseline Priority Evaluation & Ranking Comparison\nEvaluating Proposed Composite System against FIFO, Severity-Only, and Severity+Aging baseline methods."),
    nbf.v4.new_code_cell("""import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('..')

from src.baseline_comparison import BaselineComparator

df = pd.read_csv('../data/processed/cleaned_grievances.csv', low_memory=False)
sample_df = df.sample(n=min(1000, len(df)), random_state=42)

comparator = BaselineComparator()
res = comparator.compare_methods(sample_df)

print("=== BASELINE COMPARISON RESULTS ===")
print(f"High Risk Complaints in Top 10% (Proposed): {res['high_risk_in_top_proposed']}")
print(f"High Risk Complaints in Top 10% (FIFO):     {res['high_risk_in_top_fifo']}")
print(f"High Risk Prioritization Gain Ratio:       {res['high_risk_gain_ratio']}x")
print(f"Average Rank Shift for CRITICAL Complaints: +{res['avg_rank_shift_critical']:.1f} ranks higher")

print("\nSpearman Rank Correlations vs Proposed System:")
for k, v in res['spearman_correlations'].items():
    print(f"  - {k}: {v}")
""")
]

with open(os.path.join(nb_dir, "05_Priority_Evaluation.ipynb"), "w", encoding="utf-8") as f:
    nbf.write(nb5, f)
print("Generated notebooks/05_Priority_Evaluation.ipynb")
