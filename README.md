# Feature Extraction and Clustering Pipeline

This repository provides a comprehensive pipeline for feature extraction, clustering using DBSCAN and Agglomerative Clustering, attribute detection, and efficient similarity search using Pinecone. The pipeline is designed to process datasets, extract meaningful features, cluster them, detect specific attributes, store the fused features in Pinecone, and perform image retrieval based on similarity queries.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Clustering](#3-clustering)
  - [4. Attribute Detection and Fused Feature Store](#4-attribute-detection-and-fused-feature-store)
  - [5. Storing Vectors in Pinecone](#5-storing-vectors-in-pinecone)
  - [6. Image Retrieval](#6-image-retrieval)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project integrates advanced neural architectures for feature extraction, clustering algorithms like DBSCAN and Agglomerative Clustering, attribute detection mechanisms, and vector databases like Pinecone to create an efficient pipeline for image analysis and retrieval.

## Features
- **Feature Extraction**: Utilize neural network architectures to extract meaningful features from images.
- **Clustering**: Implement DBSCAN and Agglomerative Clustering algorithms to group similar data points.
- **Attribute Detection**: Detect specific attributes in images to create fused feature representations.
- **Vector Database**: Store and manage feature vectors using Pinecone for efficient similarity search.
- **Image Retrieval**: Perform similarity-based image retrieval using cosine similarity measures.

## Installation
To set up the environment, follow these steps:

1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/feature-extraction-clustering-pipeline.git
cd feature-extraction-clustering-pipeline
```

2. **Create a Virtual Environment**:
```bash
python3 -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate'
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

Ensure that the `requirements.txt` file includes all necessary packages, such as:
- torch (for PyTorch)
- tensorflow (for TensorFlow)
- scikit-learn
- matplotlib
- pinecone-client
- numpy
- pandas

## Usage

### 1. Data Preparation
Prepare your dataset by organizing your images or data points in a structured format. Ensure that the data is accessible for feature extraction.

### 2. Feature Extraction
Extract features using the desired neural network architecture:

Example:
```python
from models.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(pretrained=True)
features = extractor.extract_features(image)
```

Ensure that the image is preprocessed appropriately for the chosen model.

### 3. Clustering
Cluster the extracted features using DBSCAN or Agglomerative Clustering:

**DBSCAN**:
```python
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.5, min_samples=5).fit(features)
labels = clustering.labels_
```

**Agglomerative Clustering**:
```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5).fit(features)
labels = clustering.labels_
```

Adjust the parameters (`eps`, `min_samples`, `distance_threshold`) based on your dataset.

### 4. Attribute Detection and Fused Feature Store
Detect specific attributes in images and create fused feature representations:

```python
from models.attribute_detector import AttributeDetector

detector = AttributeDetector()
attributes = detector.detect_attributes(image)
fused_features = fuse_features(features, attributes)
```

Implement the `fuse_features` function to combine features and attributes as needed.

### 5. Storing Vectors in Pinecone
Initialize Pinecone and upload your fused feature vectors:

```python
import pinecone

pinecone.init(api_key='YOUR_API_KEY', environment='us-west1-gcp')

index = pinecone.Index('fused-feature-index')
index.upsert(items=[(str(i), fused_feature) for i, fused_feature in enumerate(fused_features)])
```

Replace `'YOUR_API_KEY'` with your actual Pinecone API key.

### 6. Image Retrieval
Perform similarity-based image retrieval using cosine similarity:

```python
query_vector = fused_features[0]  # Example query
results = index.query(queries=[query_vector], top_k=5, include_values=True)
```

## Configuration
Configure the parameters in the `config.yaml` file:

```yaml
feature_extraction:
  model: 'efficientnet-b0'  # Options: 'yolov11', 'efficientnet-b0'
  pretrained: true

clustering:
  algorithm: 'dbscan'  # Options: 'dbscan', 'agglomerative'
  eps: 0.5
  min_samples: 5

attribute_detection:
  enabled: true
  method: 'default'  # Specify the attribute detection method

pinecone:
  api_key: 'YOUR_API_KEY'
  environment: 'us-west1-gcp'
  index_name: 'fused-feature-index'
```

## Contributing
We welcome contributions! Please read our contributing guidelines for more information.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
