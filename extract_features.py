import cv2
import numpy as np
import os
import json
from insightface.app import FaceAnalysis
import re

# Initialize Face Detection and Feature Extraction Model (IR-SE50)
face_app = FaceAnalysis(name='buffalo_l')  # Uses IR-SE50 model
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Define directories
input_folder = r"path to your images to add in your database"
cropped_faces_dir = "cropped_faces"
feature_file = "features.json"
failed_images_file = "failed_images.json"   # JSON file to store images without feature extraction

# Create necessary folders if not exist
os.makedirs(cropped_faces_dir, exist_ok=True)

# Natural sorting function to ensure sequential ordering
def natural_sort_key(filename):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

# Function to detect faces, crop them, and extract feature vectors
def process_images_in_folder(input_folder, cropped_faces_dir, feature_file, failed_images_file):
    feature_dict = {}
    failed_images = []

    # List all image files in the input folder and sort naturally
    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
        key=natural_sort_key
    )

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error: Cannot load image {img_path}, deleting...")
            os.remove(img_path)  # Delete unreadable images
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for InsightFace
        faces = face_app.get(img_rgb)  # Detect faces

        if not faces:
            print(f"No face detected in {img_path}, deleting...")
            os.remove(img_path)  # Delete images with no detected face
            failed_images.append(img_name)
            continue

        img_h, img_w, _ = img.shape  # Get image dimensions

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)  # Get bounding box coordinates

            # Ensure bounding box is within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Crop the detected face
            cropped_face = img[y1:y2, x1:x2]

            # Validate cropped face is not empty
            if cropped_face.size == 0:
                print(f"Invalid face crop in {img_name}, deleting...")
                os.remove(img_path)
                failed_images.append(img_name)
                continue

            # Save cropped face image with sequential numbering
            cropped_face_filename = f"{os.path.splitext(img_name)[0]}.jpg"
            cropped_face_path = os.path.join(cropped_faces_dir, cropped_face_filename)
            cv2.imwrite(cropped_face_path, cropped_face)

            # Extract feature vector
            feature_vector = face.embedding.tolist()  # Convert numpy array to list

            if not feature_vector:
                print(f"Feature vector could not be extracted from {img_name}, deleting...")
                os.remove(img_path)  # Delete images without feature vectors
                failed_images.append(img_name)
                continue

            # Store feature vector in dictionary
            feature_dict[cropped_face_filename] = feature_vector

            print(f"Face {i+1} detected in {img_name}. Cropped image saved as {cropped_face_filename}")

    # Save all extracted feature vectors to a JSON file
    with open(feature_file, "w") as f:
        json.dump(feature_dict, f, indent=4)

    # Save failed image names to a separate JSON file
    with open(failed_images_file, "w") as f:
        json.dump(failed_images, f, indent=4)

    print(f"\nAll feature vectors saved in {feature_file}")
    print(f"Images without feature extraction saved in {failed_images_file}")

# Run the face processing function
process_images_in_folder(input_folder, cropped_faces_dir, feature_file, failed_images_file)
