import os
import glob
import pickle
import shutil
import cv2
import time
import numpy as np
import torch
from torchvision import transforms
import timm
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path

# -------------------------------
# CONFIGURATION & PATHS
# -------------------------------
# Directory containing fused features (.pkl files, dimension=1289)
fused_feature_store_dir = "fused_feature_store_attribute"
# Directory containing original dataset images
dataset_folder = "tango-cv-assessment-dataset"
# Base output folder for query results and attribute classification results
output_folder = "query_results"

# Pinecone configuration for fused features index (dimension=1289)
pinecone_api_key = "pcsk_2YxcGX_EKWZepCC8RwknJYX9Y9Tbts5y4TaoGGRt8S8wRYrjGFDrDfxHEz29UZHizWBkx"  # Replace with your API key
pinecone_index_name = "new-fused-features-index"  # Must have been created with dimension=1289

# Query parameters
top_k = 10  # Number of top neighbors to use for overall query results; we retrieve more for filtering

# Attribute configuration
attribute_labels = ['shirt', 'jacket', 'dress', 'bag', 'hat', 'sunglasses', 'scarf', 'male', 'female']
num_attr = len(attribute_labels)  # Expected to be 9
attr_threshold = 0.5  # Threshold for attribute presence

# Dimensions (for fused features)
visual_dim = 1280
fused_dim = 1289  # visual_dim + num_attr (i.e. 1280+9)

# -------------------------------
# INITIALIZE PINECONE INDEX
# -------------------------------
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# -------------------------------
# EFFICIENTNET MODEL SETUP (for query extraction)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True)
efficientnet_model.reset_classifier(0)  # Remove classification head so model returns 1280-dim embeddings
efficientnet_model.eval()
efficientnet_model.to(device)

# -------------------------------
# PREPROCESSING PIPELINE (for query image)
# -------------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# FUNCTION: LOAD FUSED FEATURES FROM .pkl
# -------------------------------
def load_fused_features_from_pkl(store_dir):
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
# HELPER FUNCTION: MAP FEATURE FILENAME TO ORIGINAL IMAGE FILENAME
# -------------------------------
def feature_to_image_filename(feature_filename):
    """
    Converts a fused feature filename to the original image filename.
    Example: "0036_c6s1_003826_00_person0_1_fused.pkl" -> "0036_c6s1_003826_00.jpg"
    Assumes the original image filename is the first 4 underscore-separated tokens plus ".jpg".
    """
    base = os.path.splitext(feature_filename)[0]  # Remove extension
    parts = base.split('_')
    if len(parts) >= 4:
        return "_".join(parts[:4]) + ".jpg"
    return feature_filename

# -------------------------------
# FUNCTION: EXTRACT ATTRIBUTE SCORES FROM A FUSED FEATURE
# -------------------------------
def extract_attribute_scores(fused_feature):
    """
    Given a fused feature vector (dimension 1289), assume the last 9 elements are attribute predictions.
    Returns a list of 9 attribute scores.
    """
    return fused_feature[-num_attr:]

# -------------------------------
# FUNCTION: COPY IMAGES BASED ON FILENAMES
# -------------------------------
def copy_images(file_names, dest_folder):
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    for fname in file_names:
        img_filename = feature_to_image_filename(fname)
        src = os.path.join(dataset_folder, img_filename)
        if os.path.exists(src):
            shutil.copy(src, dest_folder)
        else:
            print(f"Source image {src} not found.")
    print(f"Copied {len(file_names)} images to {dest_folder}")

# -------------------------------
# FUNCTION: EXTRACT VISUAL FEATURE FROM QUERY IMAGE
# -------------------------------
def extract_visual_feature(query_image_path):
    img = cv2.imread(query_image_path)
    if img is None:
        raise ValueError(f"Query image '{query_image_path}' could not be loaded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        visual_feature = efficientnet_model(input_tensor).cpu().numpy().flatten()
    return visual_feature

# -------------------------------
# FUNCTION: QUERY PINECONE INDEX
# -------------------------------
def query_pinecone_index(index, query_feature, top_k=top_k):
    query_vector = np.expand_dims(query_feature.astype("float32"), axis=0)
    results = index.query(vector=query_vector.tolist(), top_k=top_k)
    return results.get("matches", [])

# -------------------------------
# MAIN QUERY FUNCTION WITH ATTRIBUTE FILTERING & SEPARATE FOLDERS
# -------------------------------
def main_query(query_image_path, desired_attributes):
    # Load local fused features for attribute lookup
    local_features, local_file_names = load_fused_features_from_pkl(fused_feature_store_dir)
    if local_features is None:
        print("No local fused features loaded.")
        return

    # Process query image: extract 1280-dim visual feature and generate dummy attributes (length=9)
    query_visual = extract_visual_feature(query_image_path)
    dummy_attr = np.random.rand(fused_dim - visual_dim)  # dummy attribute predictions
    query_fused_feature = np.concatenate([query_visual, dummy_attr])
    print(f"Query fused feature dimension: {query_fused_feature.shape[0]}")

    # Query the Pinecone index for extra neighbors (to allow attribute filtering)
    matches = query_pinecone_index(index, query_fused_feature, top_k=top_k*3)
    print(f"Retrieved {len(matches)} candidate matches from Pinecone.")
    
    # For each candidate, lookup its local fused feature to extract attribute scores and filter
    filtered_results = {attr: [] for attr in desired_attributes}
    overall_results = []
    for match in matches:
        fid = match['id']
        overall_results.append(fid)
        if fid in local_file_names:
            idx = local_file_names.index(fid)
            fused_feat = local_features[idx]
            attr_scores = extract_attribute_scores(fused_feat)
            for attr in desired_attributes:
                if attr in attribute_labels:
                    attr_idx = attribute_labels.index(attr)
                    if attr_scores[attr_idx] >= attr_threshold:
                        filtered_results[attr].append(fid)
        else:
            print(f"Feature ID {fid} not found in local mapping.")
        if len(overall_results) >= top_k:
            break

    # Copy overall top-k results into a folder named after the query image
    query_base = os.path.splitext(os.path.basename(query_image_path))[0]
    copy_images(overall_results, os.path.join(output_folder, query_base))

    # For each desired attribute, copy the corresponding images into separate folders
    for attr, fids in filtered_results.items():
        if fids:
            dest = os.path.join(output_folder, attr)
            copy_images(fids, dest)
        else:
            print(f"No images matched attribute '{attr}'.")
    
    return {
        'overall_nearest_neighbors': overall_results,
        'filtered_results': filtered_results,
        'matches': matches,
        'query_feature_dimension': query_fused_feature.shape[0]
    }

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # Example usage:
    query_image_path = "tango-cv-assessment-dataset/0036_c6s1_003826_00.jpg"
    # Provide desired attributes as a list; these will determine separate folders for classification
    desired_attributes = ['bag', 'sunglasses', 'male']
    main_query(query_image_path, desired_attributes)
