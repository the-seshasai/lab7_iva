import cv2
import os
import pickle
from skimage.feature import local_binary_pattern

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths for male and female datasets
male_dataset_path = "male"
female_dataset_path = "female"
output_features_path = "extracted_features"

# Ensure output directory exists
if not os.path.exists(output_features_path):
    os.makedirs(output_features_path)

def extract_geometric_features(image):
    # Placeholder for actual geometric feature extraction (landmark-based)
    return {"jaw_width": 100, "eye_distance": 50}  # Example values

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    return lbp

def process_dataset(dataset_path, label):
    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            
            # Extract features
            geo_features = extract_geometric_features(cropped_face)
            texture_features = extract_texture_features(cropped_face)
            
            # Save the extracted features
            output_filename = os.path.join(output_features_path, f"{label}_{filename.split('.')[0]}_features.pkl")
            with open(output_filename, 'wb') as f:
                pickle.dump({'geo_features': geo_features, 'texture_features': texture_features}, f)
    
    print(f"Feature extraction for {label} dataset complete.")

# Process male and female datasets
process_dataset(male_dataset_path, "male")
process_dataset(female_dataset_path, "female")
