import cv2
import numpy as np
import os

# Path to the input image
input_image_path = 'in/in2.png'

# Load the image
image = cv2.imread(input_image_path)

# Parameters for smile/frown detection
smile_threshold = 10  # Tune this value for upward/downward mouth curvature

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_mouth.xml')

# Skin color ranges in HSV for hand detection
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Sentiment counters
sentiment_count = {
    "Happy": 0,
    "Sad": 0,
    "Neutral": 0,
    "Surprised": 0
}

# Detect faces and features in the image
def analyze_facial_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect mouth inside face region
        face_region = gray[y:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(face_region, 1.7, 11)

        sentiment = "Neutral"  # Default sentiment
        for (mx, my, mw, mh) in mouths:
            mouth_center_y = my + mh // 2
            if mouth_center_y > smile_threshold:  # Downward curvature
                sentiment = "Sad"
            else:  # Upward curvature
                sentiment = "Happy"

            # Draw rectangle around the mouth
            cv2.rectangle(image, (x + mx, y + my), (x + mx + mw, y + my + mh), (255, 0, 0), 2)

        # Detect eyes for a more refined emotion classification (e.g., raised eyebrows for "sad")
        eyes = eye_cascade.detectMultiScale(face_region)

        # Update sentiment counter
        sentiment_count[sentiment] += 1

        # Annotate sentiment
        cv2.putText(image, sentiment, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Detect gestures (hands raised)
def detect_hand_gestures(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this threshold based on the size of hands
            # Get bounding box for each hand detected
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.putText(image, "Hand Gesture", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Categorize overall sentiment based on the majority sentiment
def categorize_overall_sentiment(sentiment_count):
    # Find the sentiment with the highest count
    overall_sentiment = max(sentiment_count, key=sentiment_count.get)
    return overall_sentiment

# Analyze facial features and gestures
analyzed_image = analyze_facial_features(image)
final_image = detect_hand_gestures(analyzed_image)

# Determine the overall sentiment of the crowd
overall_sentiment = categorize_overall_sentiment(sentiment_count)

# Annotate the overall sentiment on the image
cv2.putText(final_image, f"Overall Sentiment: {overall_sentiment}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

# Save and display the final image
output_image_path = 'gesture_and_facial_analysis_result_with_sentiment.jpg'
cv2.imwrite(output_image_path, final_image)

print(f"Gesture and facial analysis completed. Overall sentiment: {overall_sentiment}")
print(f"Output saved at {output_image_path}.")
