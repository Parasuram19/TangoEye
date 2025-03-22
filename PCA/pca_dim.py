import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(features, file_names, clusters):
    """
    Visualize high-dimensional embeddings using PCA and t-SNE.
    
    Parameters:
    - features: numpy array of shape (n_samples, n_features)
    - file_names: list of filenames corresponding to each feature vector
    - clusters: dict mapping file names to cluster labels
    """
    # Create a cluster label array in the same order as file_names
    cluster_labels = np.array([clusters[fn] for fn in file_names])
    
    # -------------------------------
    # PCA Visualization
    # -------------------------------
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1],
                          c=cluster_labels, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title("PCA Visualization of Feature Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()
    
    # -------------------------------
    # t-SNE Visualization
    # -------------------------------
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                          c=cluster_labels, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title("t-SNE Visualization of Feature Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    # Example: load your precomputed features, file_names, and clusters.
    # For demonstration, you may load features from a file or assume they are already defined.
    # Uncomment and modify the following lines as needed:
    #
    # features = np.load("features.npy")
    # with open("file_names.txt", "r") as f:
    #     file_names = [line.strip() for line in f.readlines()]
    # with open("clusters.npy", "rb") as f:
    #     clusters = np.load(f, allow_pickle=True).item()
    
    # For this example, we'll assume the variables exist:
    # features: np.array, file_names: list, clusters: dict
    
    # Replace the following dummy code with your actual data.
    n_samples, n_features = 500, 128  # Example dimensions
    features = np.random.rand(n_samples, n_features)
    file_names = [f"img_{i}.jpg" for i in range(n_samples)]
    # Simulate clusters (e.g., 10 clusters and -1 for noise)
    clusters = {fn: np.random.randint(-1, 10) for fn in file_names}
    
    visualize_embeddings(features, file_names, clusters)
