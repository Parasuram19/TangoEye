import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import faiss
import timm
import torch.nn as nn

# -------------------------------
# CONFIGURATION & PATHS
# -------------------------------
dataset_folder = "tango-cv-assessment-dataset"   # Folder with dataset images
features_folder = "fused_feature_store_attribute"      # Optional folder to save fused features
dbscan_eps = 0.5       # DBSCAN parameter (tune as needed)
dbscan_min_samples = 3 # DBSCAN parameter (tune as needed)
k_neighbors = 5        # Number of neighbors to retrieve
attribute_labels = ['shirt', 'jacket', 'dress', 'bag', 'hat', 'sunglasses', 'scarf', 'male', 'female']

# -------------------------------
# DUMMY ATTRIBUTE CLASSIFIER
# -------------------------------
class DummyAttributeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DummyAttributeClassifier, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # For demonstration, return random probabilities in the range [0, 1]
        batch_size = x.shape[0]
        # You could also return fixed values (e.g., zeros) if preferred
        return torch.rand(batch_size, self.num_classes)

# Instead of loading a pre-trained model, we use the dummy version.
num_attr_classes = len(attribute_labels)
attribute_model = DummyAttributeClassifier(num_attr_classes)
attribute_model.eval()

# -------------------------------
# EFFICIENTNET SETUP (Visual Embedding)
# -------------------------------
# Load EfficientNet from timm and remove its classification head.
efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True)
efficientnet_model.reset_classifier(0)   # Remove the classification head to get embeddings
efficientnet_model.eval()

# -------------------------------
# PREPROCESSING PIPELINES
# -------------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# Use the same preprocessing for the attribute model.
attr_preprocess = preprocess

# -------------------------------
# FUNCTION: PROCESS IMAGE TO EXTRACT FUSED FEATURES
# -------------------------------
def process_image_fused(image_path):
    """
    Given an image path, extract visual features (via EfficientNet)
    and attribute predictions (via the dummy attribute classifier), then
    fuse them by concatenation.
    Returns: fused feature vector and the raw attribute prediction vector.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image {image_path} could not be loaded.")
    
    # Convert image from BGR to RGB for PIL compatibility
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess for visual embedding
    input_tensor = preprocess(img_rgb)
    input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        visual_feature = efficientnet_model(input_tensor).squeeze()  # shape: (d,)
    
    # Preprocess for attribute extraction (using the same pipeline here)
    attr_tensor = attr_preprocess(img_rgb)
    attr_tensor = attr_tensor.unsqueeze(0)
    with torch.no_grad():
        attr_preds = attribute_model(attr_tensor).squeeze()  # shape: (num_attributes,)
    
    # Fuse features by concatenating the visual and attribute vectors.
    fused_feature = torch.cat([visual_feature, attr_preds], dim=0).cpu().numpy()
    return fused_feature, attr_preds.cpu().numpy()

# -------------------------------
# STEP 1: EXTRACT FUSED FEATURES FROM DATASET IMAGES
# -------------------------------
fused_features = []
fused_file_names = []

for file in os.listdir(dataset_folder):
    if file.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(dataset_folder, file)
        fused_feat, _ = process_image_fused(image_path)
        fused_features.append(fused_feat)
        fused_file_names.append(file)

fused_features = np.vstack(fused_features)
print(f"Extracted fused features for {fused_features.shape[0]} images; feature dimension: {fused_features.shape[1]}")

# -------------------------------
# STEP 2: CLUSTERING WITH DBSCAN ON FUSED FEATURES
# -------------------------------
fused_features_norm = normalize(fused_features)
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
cluster_labels = dbscan.fit_predict(fused_features_norm)
clusters = {fname: label for fname, label in zip(fused_file_names, cluster_labels)}
print("Cluster assignments for dataset:")
for fname, label in clusters.items():
    print(f"  {fname}: Cluster {label}")

# -------------------------------
# STEP 3: INDEXING FOR RETRIEVAL WITH FAISS ON FUSED FEATURES
# -------------------------------
fused_features_index = fused_features.astype('float32')
d_fused = fused_features_index.shape[1]
index = faiss.IndexFlatL2(d_fused)
index.add(fused_features_index)
print(f"Indexed {index.ntotal} fused feature vectors.")

def retrieve_similar_fused(query_feature, k=k_neighbors):
    """
    Retrieve the top k nearest neighbors for a given fused query feature.
    """
    query_feature = np.expand_dims(query_feature.astype('float32'), axis=0)
    distances, indices = index.search(query_feature, k)
    return distances, indices

# -------------------------------
# STEP 4: PROCESS A QUERY IMAGE AND PERFORM RETRIEVAL
# -------------------------------
# Specify the path to your query image.
query_image_path = "tango-cv-assessment-dataset/0036_c6s1_003826_00.jpg"
query_fused_feature, query_attr_preds = process_image_fused(query_image_path)

# Retrieve similar images based on the fused features.
distances, indices = retrieve_similar_fused(query_fused_feature, k=k_neighbors)
print("\nRetrieved nearest neighbors for query image (fused features):")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"Rank {i+1}: File: {fused_file_names[idx]}, Distance: {dist:.4f}")

# Determine the cluster assignment of the query image using its nearest neighbor.
nearest_neighbor_file = fused_file_names[indices[0][0]]
query_cluster = clusters.get(nearest_neighbor_file, -1)
print(f"\nThe query image is most similar to cluster: {query_cluster}")

# Threshold attribute predictions to get a list of predicted attributes.
attr_threshold = 0.5
predicted_attributes = [attribute_labels[i] for i, p in enumerate(query_attr_preds) if p > attr_threshold]
print("Predicted attributes for query image:", predicted_attributes)
