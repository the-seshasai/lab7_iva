import cv2
import os

# Load the video
video_path = 'in.mp4'
cap = cv2.VideoCapture(video_path)

# Directory to save grayscale frames
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

# Frame counter
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Save the grayscale frame
    frame_filename = os.path.join(output_dir, f'frame_{frame_num}.png')
    cv2.imwrite(frame_filename, gray_frame)
    
    frame_num += 1

cap.release()
cv2.destroyAllWindows()

print(f"Total {frame_num} frames extracted and saved in {output_dir}")
