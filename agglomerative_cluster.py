import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import faiss
import timm

# -------------------------------
# CONFIGURATION & PATHS
# -------------------------------
features_folder = 'feature_store'  # Folder with saved .npy feature files
clustering_threshold = 1.0  # Distance threshold for agglomerative clustering (tune as needed)
k_neighbors = 5             # Number of neighbors to retrieve in search

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_features(features_dir):
    """Load feature vectors from .npy files and return stacked array and filenames."""
    feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.npy')]
    features, file_names = [], []
    for file in feature_files:
        feat = np.load(file)
        features.append(feat)
        file_names.append(os.path.basename(file))
    return np.vstack(features), file_names


def process_query_image(query_image_path, preprocess, model):
    """
    Process a query image: load, convert color, preprocess,
    and extract feature embeddings using the given model.
    """
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        raise ValueError("Query image could not be loaded.")
    # Convert BGR to RGB for PIL compatibility
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        # Model returns feature embeddings since classifier head is removed.
        query_feature = model(input_tensor).squeeze().cpu().numpy()
    return query_feature


def retrieve_similar(query_feature, index, k=k_neighbors):
    """Given a query feature, search the FAISS index for top-k similar features."""
    query_feature = np.expand_dims(query_feature.astype('float32'), axis=0)
    distances, indices = index.search(query_feature, k)
    return distances, indices


def main():
    # -------------------------------
    # STEP 1: LOAD FEATURES
    # -------------------------------
    features, file_names = load_features(features_folder)
    print(f"Loaded {features.shape[0]} features of dimension {features.shape[1]}.")

    # -------------------------------
    # STEP 2: CLUSTERING WITH AGGLOMERATIVE CLUSTERING
    # -------------------------------
    # Normalize features for more meaningful distances
    features_norm = normalize(features)
    # Use AgglomerativeClustering with a distance threshold to determine clusters automatically.
    agg_cluster = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=clustering_threshold, 
        linkage='ward'  # Ward works with Euclidean distances
    )
    cluster_labels = agg_cluster.fit_predict(features_norm)
    clusters = {fname: label for fname, label in zip(file_names, cluster_labels)}
    print("Cluster assignments:")
    for fname, label in clusters.items():
        print(f"  {fname}: Cluster {label}")

    # -------------------------------
    # STEP 3: INDEXING FOR RETRIEVAL WITH FAISS
    # -------------------------------
    features_index = features.astype('float32')
    d = features_index.shape[1]  # Feature dimensionality
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(features_index)
    print(f"Indexed {faiss_index.ntotal} feature vectors.")

    # -------------------------------
    # STEP 4: SETUP THE QUERY IMAGE PROCESSING
    # -------------------------------
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load EfficientNet from timm and remove its classification head
    efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True)
    efficientnet_model.reset_classifier(0)  # Remove classifier head for embeddings
    efficientnet_model.eval()
    efficientnet_model.to(device)

    # -------------------------------
    # STEP 5: PROCESS A QUERY IMAGE AND RETRIEVE RESULTS
    # -------------------------------
    query_image_path = "tango-cv-assessment-dataset/0002_c3s1_000076_00.jpg"
    query_feature = process_query_image(query_image_path, preprocess, efficientnet_model)

    distances, indices = retrieve_similar(query_feature, faiss_index, k=k_neighbors)
    print("\nRetrieved nearest neighbors for query image:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"Rank {i+1}: File: {file_names[idx]}, Distance: {dist:.4f}")

    # Determine the cluster assignment of the query image using its nearest neighbor.
    nearest_neighbor_file = file_names[indices[0][0]]
    query_cluster = clusters.get(nearest_neighbor_file, -1)
    print(f"\nThe query image is most similar to cluster: {query_cluster}")


if __name__ == '__main__':
    main()
