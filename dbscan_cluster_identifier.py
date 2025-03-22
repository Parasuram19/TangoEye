import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import faiss
import timm

# -------------------------------
# CONFIGURATION & PATHS
# -------------------------------
features_folder = 'feature_store'  # Folder with saved .npy feature files
dbscan_eps = 1       # DBSCAN parameter (tune as needed)
dbscan_min_samples = 3  # DBSCAN parameter (tune as needed)
k_neighbors = 5         # Number of neighbors to retrieve in search

# -------------------------------
# STEP 1: LOAD FEATURES
# -------------------------------
def load_features(features_dir):
    feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.npy')]
    features, file_names = [], []
    for file in feature_files:
        feat = np.load(file)
        features.append(feat)
        file_names.append(os.path.basename(file))
    # Stack into a matrix: shape (n_samples, n_features)
    return np.vstack(features), file_names

features, file_names = load_features(features_folder)
print(f"Loaded {features.shape[0]} features of dimension {features.shape[1]}.")

# -------------------------------
# STEP 2: CLUSTERING WITH DBSCAN
# -------------------------------
features_norm = normalize(features)
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
cluster_labels = dbscan.fit_predict(features_norm)
clusters = {fname: label for fname, label in zip(file_names, cluster_labels)}
print("Cluster assignments:")
for fname, label in clusters.items():
    print(f"  {fname}: Cluster {label}")

# -------------------------------
# STEP 3: INDEXING FOR RETRIEVAL WITH FAISS
# -------------------------------
features_index = features.astype('float32')
d = features_index.shape[1]  # Feature dimensionality
index = faiss.IndexFlatL2(d)
index.add(features_index)
print(f"Indexed {index.ntotal} feature vectors.")

# -------------------------------
# STEP 4: RETRIEVAL FUNCTION
# -------------------------------
def retrieve_similar(query_feature, k=k_neighbors):
    query_feature = np.expand_dims(query_feature.astype('float32'), axis=0)
    distances, indices = index.search(query_feature, k)
    return distances, indices

# -------------------------------
# STEP 5: PROCESSING THE QUERY IMAGE (OPTION 1)
# -------------------------------
# Define the preprocessing pipeline (ensure it matches the one used to generate dataset features)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load an EfficientNet model via timm and reset its classifier head
efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True)
efficientnet_model.reset_classifier(0)   # Remove the classification head to obtain feature embeddings
efficientnet_model.eval()

def process_query_image(query_image_path):
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        raise ValueError("Query image could not be loaded.")
    # If needed, perform person detection (e.g., using YOLO) to crop the image.
    # Here we assume the query image is already the region of interest.
    
    # Convert BGR to RGB for PIL compatibility
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    input_tensor = preprocess(query_img)
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        # Using Option 1: call the model's forward method directly.
        # Since the classifier is removed, the model returns feature embeddings.
        query_feature = efficientnet_model(input_tensor).squeeze().cpu().numpy()
    return query_feature

# -------------------------------
# STEP 6: USE THE QUERY IMAGE FOR RETRIEVAL & CLUSTER IDENTIFICATION
# -------------------------------
# Specify the path to your query image
query_image_path = "tango-cv-assessment-dataset/0002_c3s1_000076_00.jpg"
query_feature = process_query_image(query_image_path)

# Retrieve similar images
distances, indices = retrieve_similar(query_feature, k=k_neighbors)
print("\nRetrieved nearest neighbors for query image:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"Rank {i+1}: File: {file_names[idx]}, Distance: {dist:.4f}")

# Determine the cluster assignment of the query image.
# A simple approach is to use the cluster label of the nearest neighbor.
nearest_neighbor_file = file_names[indices[0][0]]
query_cluster = clusters.get(nearest_neighbor_file, -1)
print(f"\nThe query image is most similar to cluster: {query_cluster}")
