# Country Socioeconomic Clustering

Unsupervised machine learning project that groups countries by socioeconomic
indicators using K-Means clustering in Python. The goal is to identify natural
groupings of countries (e.g., developed, developing, and least-developed) based
on health, income, and economic variables.

## Dataset

The analysis uses `Country-data.csv`, containing **167 countries** and **9
numeric indicators**:

| Variable      | Description                                    |
|---------------|------------------------------------------------|
| `child_mort`  | Deaths of children under 5 per 1,000 births    |
| `exports`     | Exports of goods and services (% of GDP)       |
| `health`      | Total health spending (% of GDP)               |
| `imports`     | Imports of goods and services (% of GDP)       |
| `income`      | Net income per person                          |
| `inflation`   | Annual inflation rate                          |
| `life_expec`  | Life expectancy                                |
| `total_fer`   | Fertility rate (children per woman)            |
| `gdpp`        | GDP per capita                                 |

## Methodology

1. **Exploratory analysis** — distribution histograms and a correlation matrix
   of all numeric variables.
2. **Preprocessing** — dropped the `country` label and standardized all features
   with `StandardScaler` so each variable contributes equally.
3. **K-Means** — used the **elbow method** (k = 1–10) and the **silhouette
   score** (k = 2–6) to choose the number of clusters. Trained the final model
   with **k = 3**.
4. **Validation** — evaluated with Silhouette, Davies-Bouldin, and
   Calinski-Harabasz scores.
5. **Comparison models** — Hierarchical (Agglomerative) clustering and DBSCAN
   were run as benchmarks against K-Means.

## Results

The K-Means model (k = 3) produced three clear socioeconomic groups:

| Cluster | Countries | Child mort. | Income  | GDP per capita | Life expec. | Profile                   |
|---------|-----------|-------------|---------|----------------|-------------|---------------------------|
| 0       | 84        | 21.9        | 12,306  | 6,486          | 72.8        | Developing / middle-income |
| 1       | 47        | 93.0        | 3,942   | 1,922          | 59.2        | Least developed / low-income |
| 2       | 36        | 5.0         | 45,672  | 42,494         | 80.1        | Developed / high-income    |

**Model comparison:**

| Model                    | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------------------------|----------|------------|----------------|-------------------|
| K-Means                  | 3        | 0.283      | 1.277          | 66.24             |
| Agglomerative (Ward)     | 4        | 0.25       | 1.08           | 48.26             |
| DBSCAN (eps=1.325)       | 2 (+45 outliers) | 0.21 | 2.64           | 22.91             |

K-Means gave the best overall separation (highest Calinski-Harabasz), producing
interpretable groups aligned with development levels.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, StandardScaler, metrics)
- matplotlib, seaborn
- scipy (dendrogram)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/fernandoriosgz/country-socioeconomic-clustering.git
   cd country-socioeconomic-clustering

