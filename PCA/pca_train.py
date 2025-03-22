import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from pathlib import Path

# -------------------------------
# CONFIGURATION & PATHS
# -------------------------------
# Directory containing fused features (assumed .pkl files, dimension = 1289)
fused_feature_store_dir = "fused_feature_store_attribute"  
# You can switch to pseudo-fused feature store if needed:
# pseudo_feature_store_dir = "pseudo_fused_feature_store_attribute"

# -------------------------------
# FUNCTIONS TO LOAD FUSED FEATURES
# -------------------------------
def load_fused_features_from_pkl(store_dir):
    """
    Load fused features from .pkl files. Each file should contain a dict with key 'feature' 
    (or the feature vector itself) and returns:
      - features: NumPy array of shape (n_samples, feature_dimension)
      - file_names: list of filenames (for mapping/identification)
    """
    files = sorted(glob.glob(os.path.join(store_dir, "*.pkl")))
    features = []
    file_names = []
    for f in files:
        try:
            with open(f, "rb") as fp:
                data = pickle.load(fp)
                feat = data["feature"] if isinstance(data, dict) and "feature" in data else data
                features.append(feat)
                file_names.append(os.path.basename(f))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    if features:
        return np.vstack(features), file_names
    return None, None

# -------------------------------
# VISUALIZATION FUNCTION
# -------------------------------
def visualize_clusters(features, cluster_labels, method="pca"):
    """
    Reduce high-dimensional features to 2D using PCA or t-SNE and visualize clusters.
    :param features: NumPy array of shape (n_samples, n_features)
    :param cluster_labels: Array of cluster labels for each feature
    :param method: "pca" or "tsne" to select the dimensionality reduction technique
    """
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    features_2d = reducer.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=cluster_labels, cmap="tab20", alpha=0.7)
    plt.title(f"Clusters Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()

# -------------------------------
# MAIN EXECUTION FOR VISUALIZATION
# -------------------------------
def main_visualization():
    # Load fused features from the specified store directory
    features, file_names = load_fused_features_from_pkl(fused_feature_store_dir)
    if features is None:
        print("No fused features loaded.")
        return
    
    print(f"Loaded {features.shape[0]} features of dimension {features.shape[1]}.")
    
    # Normalize features for better clustering
    features_norm = normalize(features)
    
    # Cluster features using AgglomerativeClustering; adjust distance_threshold as needed
    clustering_threshold = 1.0
    agg_cluster = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=clustering_threshold, 
        linkage='ward'
    )
    cluster_labels = agg_cluster.fit_predict(features_norm)
    
    print(f"Number of clusters found: {len(set(cluster_labels))}")
    
    # Visualize clusters using PCA
    visualize_clusters(features, cluster_labels, method="pca")
    # Visualize clusters using t-SNE
    visualize_clusters(features, cluster_labels, method="tsne")

if __name__ == "__main__":
    main_visualization()
