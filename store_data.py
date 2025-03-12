import os
import json
import mysql.connector
import re

# Database Connection
conn = mysql.connector.connect(
    host="localhost",  
    user="root",       
    password="your password",  # Change as needed
    database="your database name"  # Change as needed
)
cursor = conn.cursor()

# Create MySQL Table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS IMAGES(
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    image_path VARCHAR(255),
    feature_vector JSON
)
''')
conn.commit()

# File Paths
image_folder = r"your cropped face image folder"  # Folder containing images
json_file = r"your feature.json"      # JSON file with embeddings
txt_file = r"your names.txt"           # TXT file with names         

# ✅ **Step 1: Read Names Sequentially from TXT File**
with open(txt_file, "r") as f:
    names = [line.strip() for line in f.readlines()]  

# ✅ **Step 2: Load Feature Vectors from JSON**
with open(json_file, "r") as f:
    feature_vectors = json.load(f)

# ✅ **Step 3: Natural Sorting Function for Correct Order**
def natural_sort_key(filename):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

# ✅ **Step 4: Get List of Image Files (Sorted Perfectly)**
image_files = sorted(os.listdir(image_folder), key=natural_sort_key)  

# ✅ **Step 5: Sequentially Insert Each Record**
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)  

    # Ensure we have a corresponding name and feature vector
    if idx < len(names):
        name = names[idx]  # Get the corresponding name

        # Extract the key from JSON based on image name (without extension)
        # image_key = os.path.splitext(image_file)[0]  # Remove '.jpg' extension
        feature_vector = feature_vectors.get(image_file, [])  # Get the feature vector
        print(image_file)

        if not feature_vector:
            print(f"⚠ No feature vector found for {image_file}, skipping...")
            continue

        # Convert feature vector to JSON format for MySQL
        feature_vector_json = json.dumps(feature_vector)

        # Insert into MySQL
        sql = "INSERT INTO IMAGES (name, image_path, feature_vector) VALUES (%s, %s, %s)"
        values = (name, image_path, feature_vector_json)

        cursor.execute(sql, values)
        conn.commit()  # ✅ Commit after every insert to ensure sequential uploading

        print(f"✅ Uploaded: {image_file} -> Name: {name} -> Features: {len(feature_vector)} values")

    else:
        print(f"⚠ Skipping {image_file}, no matching name found!")

# Close Database Connection
cursor.close()
conn.close()

print("✅ All data uploaded sequentially into MySQL!")
