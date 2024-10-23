import cv2
import os
import pickle
from skimage.feature import local_binary_pattern

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths for saved features and testing
test_image_path = "p.jpg"  # The new image to be classified
features_path = "extracted_features"

def extract_geometric_features(image):
    # Placeholder for actual geometric feature extraction (landmark-based)
    return {"jaw_width": 100, "eye_distance": 50}  # Example values

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    return lbp

# Function to extract features for testing
def extract_test_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cropped_face = image[y:y+h, x:x+w]
        geo_features = extract_geometric_features(cropped_face)
        texture_features = extract_texture_features(cropped_face)
        return geo_features, texture_features
    return None, None  # In case no face is detected

# Function to classify gender by comparing test features with male and female features
def classify_gender(test_geo, test_texture):
    male_score = 0
    female_score = 0
    
    for filename in os.listdir(features_path):
        with open(os.path.join(features_path, filename), 'rb') as f:
            features = pickle.load(f)
            geo_features = features['geo_features']
            texture_features = features['texture_features']
            
            # Compare geometric features
            if geo_features["jaw_width"] == test_geo["jaw_width"]:
                if "male" in filename:
                    male_score += 1
                else:
                    female_score += 1
            
            # Compare texture features (you can implement similarity measures for LBP)
            # Placeholder comparison
            if len(texture_features) == len(test_texture):
                if "male" in filename:
                    male_score += 1
                else:
                    female_score += 1

    return "Male" if male_score > female_score else "Female"

# Load and process the test image
test_img = cv2.imread(test_image_path)
test_geo, test_texture = extract_test_features(test_img)

# Ensure arrays are not None before checking conditions
if test_geo is not None and test_texture is not None:
    gender = classify_gender(test_geo, test_texture)
    print(f"The predicted gender is: {gender}")
else:
    print("No face detected in the test image.")
