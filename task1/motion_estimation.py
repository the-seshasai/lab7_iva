import cv2
import os
import numpy as np

# Directory where grayscale frames are saved
frame_dir = 'frames'

# Get the list of frame filenames
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))

# Directories for saving output
highlight_dir = 'highlighted_motion_frames'
event_dir = 'annotated_event_frames'
os.makedirs(highlight_dir, exist_ok=True)
os.makedirs(event_dir, exist_ok=True)

# Parameters
motion_threshold = 30  # For pixel differences
event_threshold = 50000  # For histogram-based event detection
fps = 30  # Example FPS value, adjust based on your video

# Initialize variables
previous_frame = None
previous_hist = None
motion_estimation = []
event_count = 0
timestamps = []

# Process each frame for motion estimation and event detection
for i, frame_file in enumerate(frame_files):
    # Load the grayscale frame
    frame_path = os.path.join(frame_dir, frame_file)
    gray_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    if previous_frame is None:
        previous_frame = gray_frame
        previous_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        continue

    # 1. Compute absolute difference between the current and previous frame (for motion highlighting)
    diff_frame = cv2.absdiff(previous_frame, gray_frame)

    # Threshold the difference to highlight moving regions
    _, thresh_frame = cv2.threshold(diff_frame, motion_threshold, 255, cv2.THRESH_BINARY)

    # Convert the grayscale frame to color to highlight moving regions
    color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    color_frame[thresh_frame > 0] = [0, 0, 255]  # Highlight moving regions in red

    # Save the highlighted motion frame
    highlighted_frame_path = os.path.join(highlight_dir, frame_file)
    cv2.imwrite(highlighted_frame_path, color_frame)

    # 2. Motion estimation (histogram-based)
    hist_curr = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist_diff = cv2.compareHist(previous_hist, hist_curr, cv2.HISTCMP_CHISQR)
    motion_estimation.append(hist_diff)

    # Update the previous frame and histogram
    previous_frame = gray_frame
    previous_hist = hist_curr

    # 3. Event detection: If histogram difference exceeds threshold
    if hist_diff > event_threshold:
        event_count += 1

        # Calculate timestamp for the event (in seconds)
        timestamp = (i + 1) / fps
        timestamps.append(timestamp)

        # Annotate the frame with event details
        cv2.putText(color_frame, f"Event {event_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(color_frame, f"Timestamp: {timestamp:.2f} sec", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the annotated event frame
        annotated_frame_path = os.path.join(event_dir, f'event_{event_count}.png')
        cv2.imwrite(annotated_frame_path, color_frame)

# Save motion estimation values and timestamps
np.savetxt('motion_estimation.txt', motion_estimation)
np.savetxt('event_timestamps.txt', timestamps)

print(f"Motion estimation values saved to 'motion_estimation.txt'.")
print(f"Highlighted motion frames saved in '{highlight_dir}'.")
print(f"Annotated event frames saved in '{event_dir}'.")
print(f"Total {event_count} significant events detected.")
