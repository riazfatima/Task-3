import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

SEARCH_EXTENSIONS = ['.csv', '.xlsx']
REQUIRED_COLUMNS = ['CustomerID', 'InvoiceNo', 'Quantity', 'UnitPrice']
DEFAULT_DATAFILES = [Path(r'c:\Users\NEW LAPTOP CITY\.vscode\Internship Projects\Project 3\Online Retail.xlsx'), Path('Online Retail.xlsx')]


def find_dataset(candidate_path: Path | None = None) -> Path:
    """Locate the dataset by explicit path or by searching common repository folders."""
    if candidate_path:
        if candidate_path.exists():
            return candidate_path
        raise FileNotFoundError(f'Dataset not found at specified path: {candidate_path}')

    for file_path in DEFAULT_DATAFILES:
        if file_path.exists():
            return file_path

    search_dirs = [Path('.'), Path('data')]
    files = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        files.extend(
            sorted(
                [path for path in search_dir.iterdir() if path.suffix.lower() in SEARCH_EXTENSIONS]
            )
        )

    if not files:
        raise FileNotFoundError(
            'No dataset found. Place a .csv or .xlsx file in the repository root or data/ directory, or pass --data <path>.'
        )

    return files[0]


def load_retail_data(path: Path) -> pd.DataFrame:
    """Load the dataset from CSV or Excel into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')

    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() == '.xlsx':
        return pd.read_excel(path)

    raise ValueError(f'Unsupported data format: {path.suffix}')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and clean the transaction dataset."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    df = df.copy()
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df.dropna(axis=0, how='all').reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Scale numeric features and return the matrix used for clustering."""
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        raise ValueError('The dataset must contain numeric feature columns for clustering.')

    transformer = ColumnTransformer([('num', StandardScaler(), numeric_columns)], remainder='drop')
    X = transformer.fit_transform(df)
    return X, numeric_columns


def run_pca(X: np.ndarray, n_components: int = 3):
    """Reduce dimensionality with PCA for visualization and variance reporting."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca


def evaluate_kmeans_and_extract_metrics(X: np.ndarray, min_k: int = 2, max_k: int = 8):
    """Compute inertia and silhouette scores, and select the model for k=3."""
    inertias = []
    silhouettes = []
    selected_model = None
    best_k = 3

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels) if k > 1 else np.nan)

        if k == best_k:
            selected_model = model

    if selected_model is None:
        raise ValueError(f'Unable to select the enforced cluster count k={best_k}.')

    return inertias, silhouettes, best_k, selected_model


def display_cluster_metrics(inertias, silhouettes, min_k=2, save_path: Path | None = None, show: bool = True):
    """Draw and optionally save the elbow and silhouette evaluation plot."""
    ks = list(range(min_k, min_k + len(inertias)))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(ks, inertias, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
    axs[0].set_title('Elbow Method (Inertia)', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Number of Clusters (k)', fontsize=10)
    axs[0].set_ylabel('Sum of Squared Distances (Inertia)', fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[1].plot(ks, silhouettes, marker='o', linestyle='-', color='#e67e22', linewidth=2)
    axs[1].set_title('Silhouette Score Analysis', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Number of Clusters (k)', fontsize=10)
    axs[1].set_ylabel('Average Silhouette Coefficient', fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def save_cluster_plot(result_df: pd.DataFrame, explained: np.ndarray, personas: dict, output_path: Path, show: bool = True):
    """Save or show the PCA cluster projection plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for idx, cluster_num in enumerate(sorted(personas.keys())):
        cluster_data = result_df[result_df['Cluster'] == cluster_num]
        ax.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            label=personas[cluster_num],
            c=colors[idx % len(colors)],
            alpha=0.75,
            edgecolors='w',
            s=45,
        )

    ax.set_title('Customer Segments in Low-Dimensional PCA Space', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'Principal Component 1 ({explained[0] * 100:.2f}% Variance)', fontsize=10)
    ax.set_ylabel(f'Principal Component 2 ({explained[1] * 100:.2f}% Variance)', fontsize=10)
    ax.legend(title='Strategic Enterprise Personas', loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def assign_personas(summary: pd.DataFrame):
    """Map the clusters to strategic persona labels for interpretation."""
    total_spend_column = next((col for col in summary.columns if 'spend' in col.lower()), None)
    if total_spend_column is None:
        return {
            0: 'CLUSTER C: LOW-ACTIVITY CHURN RISK',
            1: 'CLUSTER B: MID-TIER EXPLORERS',
            2: 'CLUSTER A: HIGH-VALUE ENGAGERS',
        }

    sorted_clusters = summary[total_spend_column].sort_values(ascending=False).index.tolist()
    return {
        sorted_clusters[0]: 'CLUSTER A: HIGH-VALUE ENGAGERS',
        sorted_clusters[1]: 'CLUSTER B: MID-TIER EXPLORERS',
        sorted_clusters[2]: 'CLUSTER C: LOW-ACTIVITY CHURN RISK',
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Customer segmentation for Project 3 using KMeans and PCA.')
    parser.add_argument('--data', type=Path, help='Path to the dataset (.csv or .xlsx).')
    parser.add_argument('--output', type=Path, default=Path('outputs'), help='Directory to save charts and summaries.')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots interactively.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = find_dataset(args.data)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading dataset: {dataset_path}')
    print(f'Saving outputs to: {output_dir.resolve()}')

    df = load_retail_data(dataset_path)
    df = clean_data(df)
    if df.empty:
        raise ValueError('The dataset contains no usable rows after cleaning.')

    print('Aggregating transactions to profile individual customers...')
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    customer_df = (
        df.groupby('CustomerID')
        .agg({'InvoiceNo': 'nunique', 'TotalSpend': 'sum', 'Quantity': 'sum'})
        .rename(columns={'InvoiceNo': 'Frequency'})
        .reset_index()
    )

    clustering_df = customer_df.drop(columns=['CustomerID'], errors='ignore')
    X, numeric_columns = build_feature_matrix(clustering_df)

    X_pca, pca_model = run_pca(X, n_components=3)
    explained = pca_model.explained_variance_ratio_
    print('PCA explained variance ratios:', np.round(explained, 4).tolist())

    inertias, silhouettes, best_k, best_model = evaluate_kmeans_and_extract_metrics(X, min_k=2, max_k=8)
    print(f'Target tier framework applied: Enforced k = {best_k}')

    display_cluster_metrics(
        inertias,
        silhouettes,
        min_k=2,
        save_path=output_dir / 'cluster_metrics.png',
        show=not args.no_show,
    )

    cluster_labels = best_model.predict(X)
    result_df = customer_df.copy()
    result_df['Cluster'] = cluster_labels
    result_df['PC1'] = X_pca[:, 0]
    result_df['PC2'] = X_pca[:, 1]
    result_df['PC3'] = X_pca[:, 2]

    summary = result_df.groupby('Cluster')[numeric_columns].median()
    personas = assign_personas(summary)

    print('\nCluster segment summary (Medians):')
    print(summary)
    print('\nAssigned Corporate Personas:')
    for cluster_id, persona_name in personas.items():
        print(f'  Cluster {cluster_id} -> {persona_name}')

    summary.to_csv(output_dir / 'cluster_summary.csv')
    save_cluster_plot(
        result_df,
        explained,
        personas,
        output_dir / 'cluster_projection.png',
        show=not args.no_show,
    )

    print(f'Wrote summary CSV and charts to {output_dir.resolve()}')


if __name__ == '__main__':
    main()
