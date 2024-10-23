import cv2
import numpy as np
import os

# Directory where images are saved
image_dir = 'in'

# Skin color detection thresholds (in HSV)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Output directory for face detection results
output_dir = 'detected_faces'
os.makedirs(output_dir, exist_ok=True)

# Load images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply skin color thresholding
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours to detect faces based on skin color regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Draw bounding boxes around detected face regions
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Minimum size filter for faces
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the face detection results
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, image)

print(f"Faces detected and saved in {output_dir}")
