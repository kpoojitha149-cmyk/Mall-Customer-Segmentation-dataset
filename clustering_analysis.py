"""
Week 3 Task: Unsupervised Learning and Clustering Analysis
Dataset: Mall Customer Segmentation Data (200 customers)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. LOAD & INITIAL EXPLORATION
# ------------------------------------------------------------------
df = pd.read_csv('Mall_Customers.csv')
df = df.rename(columns={'Annual Income (k$)': 'AnnualIncome', 'Spending Score (1-100)': 'SpendingScore'})
print("Shape:", df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
print(df['Gender'].value_counts())

# ------------------------------------------------------------------
# 2. PREPROCESSING
#    Features chosen: AnnualIncome and SpendingScore are the primary
#    two-dimensional clustering features (classic for this dataset,
#    and the pair that produces the clearest, most interpretable
#    segments). Age is added later as a third dimension for an
#    extended, more realistic multivariate analysis. Gender is
#    encoded but kept out of the main distance-based clustering
#    since it is categorical and would need separate handling
#    (e.g. Gower distance) to combine fairly with continuous features;
#    it is instead used afterwards to profile the resulting clusters.
# ------------------------------------------------------------------
df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

features_2d = ['AnnualIncome', 'SpendingScore']
X2 = df[features_2d].values
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

features_3d = ['Age', 'AnnualIncome', 'SpendingScore']
X3 = df[features_3d].values
scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(X3)

print("\nScaled 2D feature means (should be ~0):", X2_scaled.mean(axis=0).round(3))
print("Scaled 2D feature stds (should be ~1):", X2_scaled.std(axis=0).round(3))

# ------------------------------------------------------------------
# 3. DETERMINING THE OPTIMAL NUMBER OF CLUSTERS (2D: Income x Spending)
#    Elbow method (inertia / within-cluster sum of squares) +
#    Silhouette score, evaluated across k = 2..10.
# ------------------------------------------------------------------
k_range = range(2, 11)
inertias = []
silhouettes = []
db_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X2_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X2_scaled, labels))
    db_scores.append(davies_bouldin_score(X2_scaled, labels))

print("\nk, inertia, silhouette, davies_bouldin")
for k, i, s, d in zip(k_range, inertias, silhouettes, db_scores):
    print(k, round(i, 2), round(s, 3), round(d, 3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(list(k_range), inertias, marker='o', color='#2980b9')
axes[0].set_title('Elbow Method: WCSS vs. Number of Clusters')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Within-Cluster Sum of Squares (Inertia)')
axes[0].axvline(5, color='red', linestyle='--', alpha=0.6, label='Chosen k = 5')
axes[0].legend()

axes[1].plot(list(k_range), silhouettes, marker='o', color='#27ae60')
axes[1].set_title('Silhouette Score vs. Number of Clusters')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Average Silhouette Score')
axes[1].axvline(5, color='red', linestyle='--', alpha=0.6, label='Chosen k = 5')
axes[1].legend()
plt.tight_layout()
plt.savefig('chart_01_elbow_silhouette.png')
plt.close()

# k=5 chosen: clear elbow bend + near-highest silhouette score (confirmed in report text)
OPTIMAL_K = 5

# ------------------------------------------------------------------
# 4. K-MEANS CLUSTERING (2D: Income x Spending) with chosen k
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=10, random_state=RANDOM_STATE)
df['Cluster_KMeans2D'] = kmeans.fit_predict(X2_scaled)
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler2.inverse_transform(centroids_scaled)

sil_avg = silhouette_score(X2_scaled, df['Cluster_KMeans2D'])
db_avg = davies_bouldin_score(X2_scaled, df['Cluster_KMeans2D'])
print(f"\nFinal K-Means (k={OPTIMAL_K}) silhouette score: {sil_avg:.3f}")
print(f"Final K-Means (k={OPTIMAL_K}) Davies-Bouldin score: {db_avg:.3f}")

palette = sns.color_palette('Set1', OPTIMAL_K)
plt.figure(figsize=(8, 6))
for c in range(OPTIMAL_K):
    sub = df[df['Cluster_KMeans2D'] == c]
    plt.scatter(sub['AnnualIncome'], sub['SpendingScore'], s=55, alpha=0.75,
                color=palette[c], label=f'Cluster {c}')
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], s=260, c='black',
            marker='X', edgecolor='white', linewidth=1.5, label='Centroids')
plt.title(f'K-Means Clustering of Mall Customers (k={OPTIMAL_K})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Segment', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart_02_kmeans_clusters.png')
plt.close()

# ------------------------------------------------------------------
# 5. SILHOUETTE PLOT (per-sample) for the chosen k
# ------------------------------------------------------------------
sample_silhouette_values = silhouette_samples(X2_scaled, df['Cluster_KMeans2D'])
plt.figure(figsize=(8, 6))
y_lower = 10
for c in range(OPTIMAL_K):
    cluster_vals = sample_silhouette_values[df['Cluster_KMeans2D'] == c]
    cluster_vals.sort()
    size = cluster_vals.shape[0]
    y_upper = y_lower + size
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                       facecolor=palette[c], edgecolor=palette[c], alpha=0.8)
    plt.text(-0.05, y_lower + 0.5 * size, str(c))
    y_lower = y_upper + 10
plt.axvline(x=sil_avg, color='red', linestyle='--', label=f'Average = {sil_avg:.3f}')
plt.title('Silhouette Plot per Cluster (k=5)')
plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster (sorted samples)')
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.savefig('chart_03_silhouette_plot.png')
plt.close()

# ------------------------------------------------------------------
# 6. HIERARCHICAL (AGGLOMERATIVE) CLUSTERING - second algorithm,
#    for comparison and to validate the K-Means segmentation.
# ------------------------------------------------------------------
linked = linkage(X2_scaled, method='ward')
plt.figure(figsize=(9, 5))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8,
           show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Customers (or merged clusters)')
plt.ylabel('Distance (Ward)')
plt.axhline(y=7, color='red', linestyle='--', label='Cut height -> 5 clusters')
plt.legend()
plt.tight_layout()
plt.savefig('chart_04_dendrogram.png')
plt.close()

agglo = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
df['Cluster_Hierarchical'] = agglo.fit_predict(X2_scaled)
sil_hier = silhouette_score(X2_scaled, df['Cluster_Hierarchical'])
print(f"\nHierarchical clustering (k={OPTIMAL_K}) silhouette score: {sil_hier:.3f}")

# Align hierarchical cluster labels to the K-Means labels for a fair visual comparison
# (cluster numbers from two different algorithms are arbitrary, so we match them by
# nearest centroid in the original feature space)
from scipy.optimize import linear_sum_assignment
hier_centroids = df.groupby('Cluster_Hierarchical')[features_2d].mean().values
cost = np.linalg.norm(centroids_original[:, None, :] - hier_centroids[None, :, :], axis=2)
row_ind, col_ind = linear_sum_assignment(cost)
remap = {col: row for row, col in zip(row_ind, col_ind)}
df['Cluster_Hierarchical_aligned'] = df['Cluster_Hierarchical'].map(remap)

agreement = (df['Cluster_KMeans2D'] == df['Cluster_Hierarchical_aligned']).mean()
print(f"Agreement between K-Means and Hierarchical cluster assignments: {agreement:.1%}")

plt.figure(figsize=(8, 6))
for c in range(OPTIMAL_K):
    sub = df[df['Cluster_Hierarchical_aligned'] == c]
    plt.scatter(sub['AnnualIncome'], sub['SpendingScore'], s=55, alpha=0.75,
                color=palette[c], label=f'Cluster {c}')
plt.title(f'Hierarchical (Agglomerative) Clustering of Mall Customers (k={OPTIMAL_K})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Segment', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart_05_hierarchical_clusters.png')
plt.close()

# ------------------------------------------------------------------
# 7. EXTENDED 3D CLUSTERING (Age, Income, Spending) for a richer,
#    more realistic segmentation than income/spending alone.
# ------------------------------------------------------------------
k_range3 = range(2, 11)
sil3 = []
for k in k_range3:
    km3 = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
    labels3 = km3.fit_predict(X3_scaled)
    sil3.append(silhouette_score(X3_scaled, labels3))
OPTIMAL_K3 = list(k_range3)[int(np.argmax(sil3))]
print(f"\nBest k for 3D (Age, Income, Spending) clustering by silhouette: {OPTIMAL_K3}")

kmeans3 = KMeans(n_clusters=OPTIMAL_K3, init='k-means++', n_init=10, random_state=RANDOM_STATE)
df['Cluster_KMeans3D'] = kmeans3.fit_predict(X3_scaled)
sil3_final = silhouette_score(X3_scaled, df['Cluster_KMeans3D'])
print(f"Final 3D K-Means (k={OPTIMAL_K3}) silhouette score: {sil3_final:.3f}")

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
palette3 = sns.color_palette('Set2', OPTIMAL_K3)
for c in range(OPTIMAL_K3):
    sub = df[df['Cluster_KMeans3D'] == c]
    ax.scatter(sub['Age'], sub['AnnualIncome'], sub['SpendingScore'],
               s=45, alpha=0.8, color=palette3[c], label=f'Cluster {c}')
ax.set_title(f'3D K-Means Clustering: Age, Income, Spending Score (k={OPTIMAL_K3})')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score')
ax.legend(title='Segment', bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig('chart_06_3d_clusters.png')
plt.close()

# ------------------------------------------------------------------
# 8. CLUSTER PROFILING (2D K-Means result, the primary segmentation
#    used for business interpretation)
# ------------------------------------------------------------------
profile = df.groupby('Cluster_KMeans2D').agg(
    Count=('CustomerID', 'count'),
    Avg_Age=('Age', 'mean'),
    Avg_Income=('AnnualIncome', 'mean'),
    Avg_Spending=('SpendingScore', 'mean'),
    Pct_Female=('Gender_encoded', 'mean'),
)
profile['Pct_Female'] = (profile['Pct_Female'] * 100).round(1)
profile[['Avg_Age', 'Avg_Income', 'Avg_Spending']] = profile[['Avg_Age', 'Avg_Income', 'Avg_Spending']].round(1)
print("\nCluster profile (2D K-Means):\n", profile)

plt.figure(figsize=(9, 5))
profile_melt = profile[['Avg_Age', 'Avg_Income', 'Avg_Spending']].reset_index().melt(
    id_vars='Cluster_KMeans2D', var_name='Metric', value_name='Value')
sns.barplot(data=profile_melt, x='Cluster_KMeans2D', y='Value', hue='Metric', palette='Set2')
plt.title('Cluster Profiles: Average Age, Income, and Spending Score')
plt.xlabel('Cluster')
plt.ylabel('Average Value')
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig('chart_07_cluster_profiles.png')
plt.close()

plt.figure(figsize=(7, 4.5))
gender_ct = pd.crosstab(df['Cluster_KMeans2D'], df['Gender'], normalize='index') * 100
gender_ct.plot(kind='bar', stacked=True, color=['#3498db', '#e84393'], ax=plt.gca())
plt.title('Gender Composition by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Percentage of Cluster')
plt.xticks(rotation=0)
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig('chart_08_gender_by_cluster.png')
plt.close()

# ------------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------------
df.to_csv('mall_customers_clustered.csv', index=False)
profile.to_csv('cluster_profile_summary.csv')

print("\nAll charts and output files saved. Clustering analysis complete.")
