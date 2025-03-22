import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import timm
from torchvision import transforms

# --- Setup directories ---
dataset_folder = "tango-cv-assessment-dataset"          # folder containing your images
features_folder = "feature_store"   # folder to save the feature files
os.makedirs(features_folder, exist_ok=True)

# --- Load Models ---
# Load YOLO model for detection
yolo_model = YOLO("yolo11n.pt")  # or "path/to/best.pt" for your custom model

# Load EfficientNet (using timm) for feature extraction
efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True)
efficientnet_model.eval()  # set model to evaluation mode

# --- Preprocessing for EfficientNet ---
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust size if necessary
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Process each image in the dataset ---
for img_name in os.listdir(dataset_folder):
    img_path = os.path.join(dataset_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue  # skip if image can't be loaded

    # --- Person Detection using YOLO ---
    results = yolo_model(image)
    
    # Iterate over detection results (for multiple persons per image)
    for idx, result in enumerate(results):
        # Assuming the detection result includes bounding boxes and class names
        for i, box in enumerate(result.boxes.xyxy):
            # For this example, assume result.boxes.cls holds the class id
            class_id = int(result.boxes.cls[i])
            # Check if detected object is a person (class id may vary based on model configuration)
            # Typically, 'person' is indexed as 0 in common datasets.
            if class_id != 0:
                continue

            # Get coordinates from the bounding box and convert to integer
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue  # Skip if crop is invalid

            # --- Feature Extraction using EfficientNet ---
            input_tensor = preprocess(cropped)
            input_tensor = input_tensor.unsqueeze(0)  # add batch dimension

            with torch.no_grad():
                features = efficientnet_model.forward_features(input_tensor)
                # Optionally, you can flatten the features if needed
                features = features.mean([2, 3]).squeeze().cpu().numpy()

            # --- Save the feature vector ---
            feature_filename = f"{os.path.splitext(img_name)[0]}_person{idx}_{i}.npy"
            feature_path = os.path.join(features_folder, feature_filename)
            np.save(feature_path, features)

            print(f"Saved features for {feature_filename}")

