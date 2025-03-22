import os
import glob
import time
import numpy as np
import pickle
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path

# -------------------------------
# CONFIGURATION & PATHS
# -------------------------------
# Directories for your saved features:
feature_store_dir = "feature_store"                     # Visual features stored as .npy
fused_feature_store_dir = "fused_feature_store_attribute" # Fused features stored as .pkl
pseudo_fused_dir = "pseudo_fused_feature_store_attribute"           # Pseudo-fused features stored as .npy

# Pinecone API key – replace with your actual API key
pinecone_api_key = "pcsk_2YxcGX_EKWZepCC8RwknJYX9Y9Tbts5y4TaoGGRt8S8wRYrjGFDrDfxHEz29UZHizWBkx"

# Initialize Pinecone using the new API:
pc = Pinecone(api_key=pinecone_api_key)

# Define index configurations.
# For example, if your visual features are 1280-dimensional and your fused features
# are visual (1280) + attributes (9) = 1289-dimensional, then:
index_info = [
    {"name": "new-features-store-index", "dir": feature_store_dir, "ext": ".npy", "dimension": 1280},
    {"name": "new-fused-features-index", "dir": fused_feature_store_dir, "ext": ".pkl", "dimension": 1289},
    {"name": "new-pseudo-fused-features-index", "dir": pseudo_fused_dir, "ext": ".npy", "dimension": 1289}
]

# -------------------------------
# FUNCTIONS TO LOAD FEATURES
# -------------------------------
def load_npy_features(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    features = []
    file_names = []
    for f in files:
        try:
            feat = np.load(f)
            features.append(feat)
            file_names.append(os.path.basename(f))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    if features:
        return np.vstack(features), file_names
    return None, None

def load_pkl_features(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.pkl")))
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
# FUNCTION: WAIT FOR INDEX DELETION
# -------------------------------
def wait_for_index_deletion(pc, index_name, timeout=120):
    """Wait until the index is deleted, or timeout after a certain period."""
    start = time.time()
    while index_name in pc.list_indexes():
        if time.time() - start > timeout:
            raise TimeoutError(f"Index {index_name} was not deleted within {timeout} seconds.")
        print(f"Waiting for index '{index_name}' to be deleted...")
        time.sleep(5)
    print(f"Index '{index_name}' deleted.")

# -------------------------------
# FUNCTION: CREATE OR RECREATE PINECONE INDEX
# -------------------------------
def create_or_recreate_index(pc, index_name, dimension):
    existing = pc.list_indexes()
    if index_name in existing:
        print(f"Index '{index_name}' exists. Deleting it to recreate with dimension {dimension}...")
        pc.delete_index(index_name)
        wait_for_index_deletion(pc, index_name)
    print(f"Creating index '{index_name}' with dimension {dimension} in region 'us-east-1' on AWS...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',         # Use AWS
            region='us-east-1'     # Supported region for free plan
        )
    )
    return pc.Index(index_name)

# -------------------------------
# FUNCTION: UPLOAD FEATURES IN BATCHES
# -------------------------------
def upload_features(index, features, file_names, batch_size=100):
    features = features.astype('float32')
    total = len(file_names)
    for i in range(0, total, batch_size):
        batch_ids = file_names[i:i+batch_size]
        batch_vectors = features[i:i+batch_size].tolist()
        vectors = list(zip(batch_ids, batch_vectors))
        index.upsert(vectors)
        print(f"Uploaded {min(i+batch_size, total)}/{total} features")

# -------------------------------
# MAIN EXECUTION FOR UPLOAD
# -------------------------------
def main_upload():
    for info in index_info:
        dir_path = info["dir"]
        ext = info["ext"]
        idx_name = info["name"]
        dim = info["dimension"]
        print(f"\nProcessing directory '{dir_path}' for index '{idx_name}' (dim={dim})...")
        if ext == ".npy":
            features, file_names = load_npy_features(dir_path)
        elif ext == ".pkl":
            features, file_names = load_pkl_features(dir_path)
        else:
            print(f"Unknown file extension: {ext}")
            continue

        if features is None:
            print(f"No features loaded from {dir_path}")
            continue

        print(f"Loaded {features.shape[0]} features from {dir_path}")
        index = create_or_recreate_index(pc, idx_name, dim)
        upload_features(index, features, file_names)
        stats = index.describe_index_stats()
        print(f"Index '{idx_name}' stats: {stats}")

if __name__ == "__main__":
    main_upload()
