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
import pickle
from pathlib import Path
import time
import glob
import sys

# -------------------------------
# CONFIGURATION & PATHS
# -------------------------------
dataset_folder = "tango-cv-assessment-dataset"   # Folder with dataset images
feature_store = "feature_store"                  # Folder that contains the existing encodings
fused_feature_store = "fused_feature_store_attribute"  # Folder to save fused features
dbscan_eps = 0.5                    # DBSCAN parameter (tune as needed)
dbscan_min_samples = 3              # DBSCAN parameter (tune as needed)
k_neighbors = 5                     # Number of neighbors to retrieve
attribute_labels = ['shirt', 'jacket', 'dress', 'bag', 'hat', 'sunglasses', 'scarf', 'male', 'female']

# -------------------------------
# HELPER FUNCTIONS FOR FEATURE LOADING
# -------------------------------
def check_feature_store():
    """Check feature store format and path validity"""
    print(f"Checking feature store: {feature_store}")
    
    if not os.path.exists(feature_store):
        print(f"ERROR: Feature store path '{feature_store}' does not exist!")
        return False
        
    if os.path.isdir(feature_store):
        pkl_files = glob.glob(os.path.join(feature_store, "*.pkl"))
        if not pkl_files:
            npy_files = glob.glob(os.path.join(feature_store, "*.npy"))
            if not npy_files:
                print(f"ERROR: No .pkl or .npy files found in {feature_store}")
                return False
            else:
                print(f"Found {len(npy_files)} .npy files in {feature_store}")
        else:
            print(f"Found {len(pkl_files)} .pkl files in {feature_store}")
    else:
        # Single file
        if not (feature_store.endswith('.pkl') or feature_store.endswith('.npy')):
            print(f"ERROR: Feature store file '{feature_store}' is not a .pkl or .npy file!")
            return False
            
    return True

def load_existing_features():
    """Load existing EfficientNet features from the feature store.
    Handles multiple possible formats.
    """
    if not check_feature_store():
        raise ValueError(f"Invalid feature store: {feature_store}")
    
    features = []
    file_names = []
    
    try:
        # Check if feature store is a directory or a single file
        if os.path.isdir(feature_store):
            # Try loading .pkl files
            pkl_files = glob.glob(os.path.join(feature_store, "*.pkl"))
            
            if pkl_files:
                for file_path in pkl_files:
                    try:
                        with open(file_path, 'rb') as f:
                            feature_data = pickle.load(f)
                            
                            # Handle different possible formats
                            if isinstance(feature_data, dict):
                                if 'feature' in feature_data and 'filename' in feature_data:
                                    features.append(feature_data['feature'])
                                    file_names.append(feature_data['filename'])
                                elif len(feature_data) == 1:
                                    # Single entry dict
                                    key = list(feature_data.keys())[0]
                                    features.append(feature_data[key])
                                    file_names.append(os.path.basename(key))
                                else:
                                    # Dict with multiple entries (filename:feature pairs)
                                    for fname, feat in feature_data.items():
                                        features.append(feat)
                                        file_names.append(os.path.basename(fname))
                            elif isinstance(feature_data, (list, np.ndarray)):
                                # Just a feature vector
                                features.append(feature_data)
                                file_names.append(os.path.basename(file_path).replace('.pkl', ''))
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}: {e}")
            
            # If no pkl files or none loaded successfully, try npy files
            if not features:
                npy_files = glob.glob(os.path.join(feature_store, "*.npy"))
                if npy_files:
                    for file_path in npy_files:
                        try:
                            feature_data = np.load(file_path, allow_pickle=True)
                            features.append(feature_data)
                            file_names.append(os.path.basename(file_path).replace('.npy', ''))
                        except Exception as e:
                            print(f"Warning: Could not load {file_path}: {e}")
        else:
            # If it's a single file
            if feature_store.endswith('.pkl'):
                with open(feature_store, 'rb') as f:
                    all_features = pickle.load(f)
                    
                    if isinstance(all_features, dict):
                        for fname, feat in all_features.items():
                            features.append(feat)
                            file_names.append(os.path.basename(fname))
                    elif isinstance(all_features, (list, np.ndarray)):
                        # If it's just a list of features, use integers as filenames
                        features = all_features
                        file_names = [f"image_{i}.jpg" for i in range(len(features))]
            elif feature_store.endswith('.npy'):
                all_features = np.load(feature_store, allow_pickle=True)
                features = all_features
                file_names = [f"image_{i}.jpg" for i in range(len(features))]
    
    except Exception as e:
        print(f"Error loading features: {e}")
        raise
    
    if not features:
        raise ValueError(f"No features could be loaded from {feature_store}")
    
    print(f"Successfully loaded {len(features)} features")
    print(f"First feature shape: {np.array(features[0]).shape}")
    
    # Ensure all features have the same shape
    feature_lens = [len(np.array(f).flatten()) for f in features]
    if len(set(feature_lens)) > 1:
        print(f"Warning: Features have inconsistent dimensions: {set(feature_lens)}")
    
    # Reshape features to 2D array if needed
    features_array = []
    for f in features:
        f_array = np.array(f)
        if f_array.ndim == 1:
            features_array.append(f_array)
        else:
            features_array.append(f_array.flatten())
    
    return np.vstack(features_array), file_names

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
        return torch.rand(batch_size, self.num_classes)

# Create paths
Path(fused_feature_store).mkdir(exist_ok=True)

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Initialize attribute model (always needed for fused features)
num_attr_classes = len(attribute_labels)
attribute_model = DummyAttributeClassifier(num_attr_classes)
attribute_model.eval()

# -------------------------------
# FUNCTION: PROCESS IMAGE TO EXTRACT ATTRIBUTES
# -------------------------------
def extract_attributes(image_path):
    """Extract only attribute predictions from an image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Image {image_path} could not be loaded.")
            return np.random.rand(num_attr_classes)  # Return random attributes as fallback
        
        # Convert image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for attribute extraction
        attr_tensor = preprocess(img_rgb)
        attr_tensor = attr_tensor.unsqueeze(0)
        
        with torch.no_grad():
            attr_preds = attribute_model(attr_tensor).squeeze()
        
        return attr_preds.cpu().numpy()
    except Exception as e:
        print(f"Error extracting attributes from {image_path}: {e}")
        return np.random.rand(num_attr_classes)  # Return random attributes as fallback

# -------------------------------
# FUNCTION: CREATE FUSED FEATURES
# -------------------------------
def create_fused_features(visual_features, file_names):
    """Create fused features by adding attribute predictions to visual features."""
    fused_features = []
    start_time = time.time()
    
    for i, file in enumerate(file_names):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processing {i}/{len(file_names)} images. Time elapsed: {elapsed:.2f}s")
        
        # Construct full image path
        image_path = os.path.join(dataset_folder, file)
        
        # If file doesn't exist in dataset folder, try using it as a complete path
        if not os.path.exists(image_path):
            image_path = file
            if not os.path.exists(image_path):
                print(f"Warning: Image {file} not found. Using random attributes.")
                attr_preds = np.random.rand(num_attr_classes)
            else:
                # Get attributes for the image
                attr_preds = extract_attributes(image_path)
        else:
            # Get attributes for the image
            attr_preds = extract_attributes(image_path)
        
        # Combine visual features with attribute predictions
        fused_feature = np.concatenate([visual_features[i], attr_preds])
        fused_features.append(fused_feature)
        
        # Save individual fused feature
        os.makedirs(fused_feature_store, exist_ok=True)
        fused_feature_path = os.path.join(fused_feature_store, f"{os.path.splitext(os.path.basename(file))[0]}_fused.pkl")
        with open(fused_feature_path, 'wb') as f:
            pickle.dump({'feature': fused_feature, 'filename': file}, f)
    
    return np.vstack(fused_features)

# -------------------------------
# STEP 1: LOAD OR EXTRACT FEATURES
# -------------------------------
def main():
    print("Step 1: Loading or extracting features...")
    start_time = time.time()

    try:
        # Load existing visual features
        print(f"Loading existing features from: {feature_store}")
        visual_features, file_names = load_existing_features()
        print(f"Loaded features for {len(file_names)} images; feature dimension: {visual_features.shape[1]}")
        
        # Load or create fused features
        if os.path.exists(fused_feature_store):
            fused_feature_files = [f for f in os.listdir(fused_feature_store) if f.endswith('.pkl')]
            
            if len(fused_feature_files) == len(file_names):
                print("Loading existing fused features...")
                fused_features = []
                for i, file in enumerate(file_names):
                    base_name = os.path.splitext(os.path.basename(file))[0]
                    fused_feature_path = os.path.join(fused_feature_store, f"{base_name}_fused.pkl")
                    
                    if os.path.exists(fused_feature_path):
                        with open(fused_feature_path, 'rb') as f:
                            feature_data = pickle.load(f)
                            fused_features.append(feature_data['feature'])
                    else:
                        # If any are missing, create all fused features
                        print(f"Missing fused feature for {file}. Creating all fused features...")
                        fused_features = create_fused_features(visual_features, file_names)
                        break
                
                if fused_features:
                    fused_features = np.vstack(fused_features)
            else:
                # Create fused features using existing visual features
                print("Creating fused features from existing visual features...")
                fused_features = create_fused_features(visual_features, file_names)
        else:
            # Create fused features directory and generate fused features
            os.makedirs(fused_feature_store, exist_ok=True)
            print("Creating fused features from existing visual features...")
            fused_features = create_fused_features(visual_features, file_names)

        print(f"Feature processing time: {time.time() - start_time:.2f} seconds")
        
        # -------------------------------
        # STEP 2: CLUSTERING WITH DBSCAN (OPTIMIZED)
        # -------------------------------
        print("\nStep 2: Clustering with DBSCAN...")
        start_time = time.time()

        # Normalize features for better clustering
        fused_features_norm = normalize(fused_features)

        # Use regular DBSCAN for simplicity and reliability
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(fused_features_norm)

        clusters = {fname: label for fname, label in zip(file_names, cluster_labels)}

        print(f"Clustering time: {time.time() - start_time:.2f} seconds")
        
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            
        print(f"Number of clusters found: {len(unique_clusters)}")
        print(f"Number of outliers (noise): {list(cluster_labels).count(-1)}")

        # -------------------------------
        # STEP 3: INDEXING FOR RETRIEVAL WITH FAISS
        # -------------------------------
        print("\nStep 3: Creating search index...")
        start_time = time.time()

        # Create index with fused features
        fused_features_index = fused_features.astype('float32')
        d_fused = fused_features_index.shape[1]

        # For simplicity, use flat index
        index = faiss.IndexFlatL2(d_fused)
        index.add(fused_features_index)

        print(f"Indexing time: {time.time() - start_time:.2f} seconds")
        print(f"Indexed {index.ntotal} feature vectors.")
        
        # -------------------------------
        # STEP 4: EXAMPLE QUERY
        # -------------------------------
        # Look for an example image in the dataset
        example_images = []
        if os.path.exists(dataset_folder):
            example_images = [
                os.path.join(dataset_folder, f) 
                for f in os.listdir(dataset_folder) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ][:5]  # Get up to 5 example images
            
        if example_images:
            query_image_path = example_images[0]
            print(f"\nPerforming example query with: {query_image_path}")
            
            try:
                # Process query image
                query_filename = os.path.basename(query_image_path)
                
                # Find query feature in feature store if possible
                query_idx = -1
                for i, fname in enumerate(file_names):
                    if os.path.basename(fname) == query_filename:
                        query_idx = i
                        break
                
                if query_idx >= 0:
                    # We already have the feature
                    query_visual_feature = visual_features[query_idx]
                    query_attr_preds = extract_attributes(query_image_path)
                    query_fused_feature = np.concatenate([query_visual_feature, query_attr_preds])
                else:
                    # Use the first feature as example and extract attributes directly
                    print("Query image not found in feature store. Using random feature as example.")
                    query_visual_feature = visual_features[0]
                    query_attr_preds = extract_attributes(query_image_path)
                    query_fused_feature = np.concatenate([query_visual_feature, query_attr_preds])
                
                # Retrieve similar images
                query_feature = np.expand_dims(query_fused_feature.astype('float32'), axis=0)
                distances, indices = index.search(query_feature, min(k_neighbors, len(file_names)))
                
                # Get results
                nearest_neighbor_file = file_names[indices[0][0]]
                query_cluster = clusters.get(nearest_neighbor_file, -1)
                
                # Threshold attribute predictions
                attr_threshold = 0.5
                predicted_attributes = [attribute_labels[i] for i, p in enumerate(query_attr_preds) if p > attr_threshold]
                
                print("\nRetrieved nearest neighbors:")
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    print(f"Rank {i+1}: File: {file_names[idx]}, Distance: {dist:.4f}")
                
                print(f"\nThe query image belongs to cluster: {query_cluster}")
                print("Predicted attributes:", predicted_attributes)
                
            except Exception as e:
                print(f"Error processing query image: {e}")
                
        print("\nSystem ready for custom queries.")
        
        # Return important variables for external use
        return {
            'visual_features': visual_features,
            'file_names': file_names,
            'fused_features': fused_features,
            'index': index,
            'clusters': clusters,
            'cluster_labels': cluster_labels
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# -------------------------------
# FUNCTION: PROCESS QUERY IMAGE
# -------------------------------
def process_query_image(image_path, visual_features, file_names):
    """Process a query image to get its fused feature vector."""
    query_filename = os.path.basename(image_path)
    
    # Check if the query image's feature exists in our loaded features
    query_idx = -1
    for i, fname in enumerate(file_names):
        if os.path.basename(fname) == query_filename:
            query_idx = i
            break
    
    if query_idx >= 0:
        # We already have the feature
        visual_feature = visual_features[query_idx]
    else:
        print(f"Query image feature not found. Using similar filename...")
        # Try to find a feature with similar name
        best_match = None
        best_score = 0
        for i, fname in enumerate(file_names):
            base_fname = os.path.basename(fname)
            # Simple scoring based on common characters
            score = sum(1 for a, b in zip(query_filename, base_fname) if a == b)
            if score > best_score:
                best_score = score
                best_match = i
        
        if best_match is not None:
            visual_feature = visual_features[best_match]
            print(f"Using feature from {file_names[best_match]} as closest match")
        else:
            # Use first feature as fallback
            visual_feature = visual_features[0]
            print("No matching feature found. Using first feature instead.")
    
    # Extract attribute predictions
    attr_preds = extract_attributes(image_path)
    
    # Fuse features
    fused_feature = np.concatenate([visual_feature, attr_preds])
    
    return fused_feature, attr_preds

# -------------------------------
# FUNCTION: PERFORM SEARCH
# -------------------------------
def search_similar_images(query_image_path, k=5, system_vars=None):
    """Search for similar images to the query image."""
    if system_vars is None:
        print("Error: System variables not initialized. Run main() first.")
        return None
    
    visual_features = system_vars['visual_features']
    file_names = system_vars['file_names']
    index = system_vars['index']
    clusters = system_vars['clusters']
    
    try:
        print(f"\nProcessing query image: {query_image_path}")
        
        # Process query image
        query_fused_feature, query_attr_preds = process_query_image(
            query_image_path, visual_features, file_names)
        
        # Retrieve similar images
        query_feature = np.expand_dims(query_fused_feature.astype('float32'), axis=0)
        distances, indices = index.search(query_feature, min(k, len(file_names)))
        
        # Get results
        nearest_neighbor_file = file_names[indices[0][0]]
        query_cluster = clusters.get(nearest_neighbor_file, -1)
        
        # Threshold attribute predictions
        attr_threshold = 0.5
        predicted_attributes = [attribute_labels[i] for i, p in enumerate(query_attr_preds) if p > attr_threshold]
        
        print("\nRetrieved nearest neighbors:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"Rank {i+1}: File: {file_names[idx]}, Distance: {dist:.4f}")
        
        print(f"\nThe query image belongs to cluster: {query_cluster}")
        print("Predicted attributes:", predicted_attributes)
        
        return {
            'nearest_neighbors': [file_names[idx] for idx in indices[0]],
            'distances': distances[0].tolist(),
            'cluster': query_cluster,
            'attributes': predicted_attributes
        }
    
    except Exception as e:
        print(f"Error searching for similar images: {e}")
        import traceback
        traceback.print_exc()
        return None

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # Run the main function to set everything up
    system_vars = main()
    
    if system_vars:
        # Example of how to use the search function
        if len(sys.argv) > 1:
            # If image path provided as command line argument
            query_image_path = sys.argv[1]
        else:
            # Default query image
            query_image_path = "tango-cv-assessment-dataset/0036_c6s1_003826_00.jpg"
            
            # If default image doesn't exist, try to find any image in the dataset
            if not os.path.exists(query_image_path) and os.path.exists(dataset_folder):
                image_files = [
                    os.path.join(dataset_folder, f) 
                    for f in os.listdir(dataset_folder) 
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                ]
                if image_files:
                    query_image_path = image_files[0]
        
        if os.path.exists(query_image_path):
            search_similar_images(query_image_path, k=5, system_vars=system_vars)
        else:
            print(f"Query image {query_image_path} not found. Please provide a valid image path.")