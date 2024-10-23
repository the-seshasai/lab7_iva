import cv2
import dlib
import os
from skimage.feature import local_binary_pattern
import numpy as np
from skimage.filters import sobel
import matplotlib.pyplot as plt

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
            label = "male" if "male" in filename.lower() else "female"
            images.append(image)
            labels.append(label)
    return images, labels

# Preprocessing: Detect face and crop it
def preprocess_image(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        return cv2.resize(face, (200, 200))  # Normalize the face size
    return None

# Geometric feature extraction: Calculate distances between facial landmarks
def extract_geometric_features(image):
    dets = detector(image, 1)
    for det in dets:
        shape = predictor(image, det)
        jaw_width = shape.part(16).x - shape.part(0).x  # Jaw width
        eye_distance = shape.part(42).x - shape.part(39).x  # Eye distance
        nose_width = shape.part(35).x - shape.part(31).x  # Nose width
        mouth_width = shape.part(54).x - shape.part(48).x  # Mouth width
        face_height = shape.part(8).y - shape.part(24).y  # Face height (chin to forehead)
        return {
            "jaw_width": jaw_width,
            "eye_distance": eye_distance,
            "nose_width": nose_width,
            "mouth_width": mouth_width,
            "face_height": face_height
        }
    return None

# Texture-based feature extraction: LBP and Sobel edge detection
def extract_texture_features(image):
    # Local Binary Pattern (LBP)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Sobel Edge Detection
    edges = sobel(image)
    
    return {
        "lbp": lbp,
        "edges": edges
    }

# Rule-based gender classification
def classify_gender(geometric, texture):
    # Geometric rules
    if geometric["jaw_width"] > 95 and geometric["eye_distance"] > 45 and geometric["nose_width"] > 25:
        return "male"
    elif geometric["jaw_width"] < 95 and geometric["mouth_width"] < 50:
        return "female"
    else:
        # Texture-based rule: Analyze the edge density or LBP histogram
        edge_density = np.sum(texture["edges"]) / (200 * 200)  # Normalize edge sum
        if edge_density > 0.15:
            return "male"
        else:
            return "female"

# Load and preprocess the dataset
dataset_path = "data"
images, labels = load_dataset(dataset_path)
preprocessed_images = [preprocess_image(image) for image in images]
preprocessed_images = [img for img in preprocessed_images if img is not None]

# Extract features and classify gender
geometric_features = [extract_geometric_features(img) for img in preprocessed_images]
texture_features = [extract_texture_features(img) for img in preprocessed_images]

# Filter out None values from feature lists
geometric_features = [feat for feat in geometric_features if feat is not None]
texture_features = [feat for feat in texture_features if feat is not None]

# Classify gender based on features
predicted_genders = [classify_gender(geo_feat, tex_feat) for geo_feat, tex_feat in zip(geometric_features, texture_features)]

# Display results
for i, gender in enumerate(predicted_genders):
    print(f"Image {i+1}: Detected Gender: {gender}")
    print(f"  Geometric Features: {geometric_features[i]}")
    print(f"  Edge Density (Texture): {np.sum(texture_features[i]['edges']) / (200 * 200):.4f}")
    
    # Optionally display images with the extracted features
    plt.subplot(1, 2, 1)
    plt.imshow(texture_features[i]["lbp"], cmap="gray")
    plt.title(f"LBP - Image {i+1}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(texture_features[i]["edges"], cmap="gray")
    plt.title(f"Edges - Image {i+1}")
    
    plt.show()
