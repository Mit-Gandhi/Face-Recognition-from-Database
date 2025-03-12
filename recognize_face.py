import cv2
import numpy as np
import json
import mysql.connector
import faiss
from insightface.app import FaceAnalysis

# MySQL Connection Setup
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='your password',
    database='your database name'
)
cursor = conn.cursor()

# Load InsightFace Model
face_app = FaceAnalysis(name='buffalo_l')  # Uses IR-SE50
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load Face Database and Build FAISS Index
def load_face_database():
    cursor.execute('SELECT id, name, feature_vector FROM IMAGES')
    records = cursor.fetchall()

    if not records:
        return None, []

    # Convert JSON string to NumPy array
    feature_vectors = [json.loads(record[2]) for record in records]  
    feature_vectors = np.array(feature_vectors).astype('float32')

    # Normalize embeddings to unit length (for cosine similarity)
    faiss.normalize_L2(feature_vectors)

    # Use FAISS index with inner product (cosine similarity)
    index = faiss.IndexFlatIP(feature_vectors.shape[1])  
    index.add(feature_vectors)

    return index, records

# Function to recognize faces in live webcam feed
def recognize_live():
    index, records = load_face_database()
    if index is None:
        print("No faces in database")
        return

    cap = cv2.VideoCapture(0)  # Open webcam (change to 0 for built-in camera)
    threshold = 0.4  # Cosine similarity threshold (adjust as needed)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for InsightFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = face_app.get(frame_rgb)

        for face in faces:
            feature_vector = face.embedding.reshape(1, -1).astype('float32')

            # Normalize extracted feature vector
            faiss.normalize_L2(feature_vector)

            # Search in FAISS for the closest match
            similarity, index_match = index.search(feature_vector, 1)  
            best_similarity = similarity[0][0]  # Cosine similarity score
            print(f"Similarity: {best_similarity:.2f}")

            x1, y1, x2, y2 = map(int, face.bbox)

            # Determine if match is valid
            if best_similarity < threshold:  # Lower similarity means unknown
                matched_name = "Unknown"
                print("Recognized: Unknown")
            else:
                matched_id, matched_name = records[index_match[0][0]][:2]
                print(f"Recognized: {matched_name} (Similarity: {best_similarity:.2f})")

            # Draw Bounding Box and Display Closest Match Name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Prepare text to display (Name + Similarity Score)
            display_text = f"{matched_name} ({best_similarity:.2f})"

            # Display Name and Similarity Score
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Resize frame for better display
        display_width = 800  # Set a fixed width
        aspect_ratio = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * aspect_ratio)
        resized_frame = cv2.resize(frame, (display_width, display_height))

        # Show the output frame
        cv2.imshow("Live Face Recognition", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Start live face recognition
recognize_live()

# Close Database Connection
cursor.close()
conn.close()
