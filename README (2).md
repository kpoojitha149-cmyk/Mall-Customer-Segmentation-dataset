# Week 3 Task: Unsupervised Learning and Clustering Analysis

## Overview
This project applies K-Means and Hierarchical clustering to segment 200 mall
customers based on annual income and spending behavior, using the public
Mall Customer Segmentation dataset.

## Contents
| File | Description |
|---|---|
| `clustering_analysis.py` | Full pipeline: preprocessing, elbow/silhouette k-selection, K-Means, Hierarchical clustering, 3D extension, cluster profiling, and all 8 figures |
| `Mall_Customers.csv` | Original public dataset (200 rows x 5 columns) |
| `mall_customers_clustered.csv` | Dataset with cluster labels appended |
| `cluster_profile_summary.csv` | Per-cluster summary statistics |
| `chart_01...chart_08.png` | All 8 generated figures |
| `Week3_Clustering_Analysis_Report.docx` | Full written report with methodology, results, and business interpretation |

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
python clustering_analysis.py
```

## Key results
- Optimal k = 5, confirmed by both the elbow method and silhouette analysis (score 0.555)
- K-Means and Hierarchical (Ward linkage) clustering agreed on 98% of customers
- Five customer personas identified: Average/Mainstream, Premium Target
  (high income + high spending), Impulsive/Young (low income + high spending),
  Cautious Saver (high income + low spending), and Budget Conscious
  (low income + low spending)
- An extended 3-feature analysis (adding Age) was also explored as a secondary result

Full methodology, validation, and business discussion are in
`Week3_Clustering_Analysis_Report.docx`.
