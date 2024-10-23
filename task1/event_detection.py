import cv2
import os
import numpy as np

# Load motion estimation values
motion_estimation = np.loadtxt('motion_estimation.txt')

# Directory where original frames are saved
frame_dir = 'frames'

# Get the list of frame filenames
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))

# Event threshold (adjust based on experimentation)
event_threshold = 50000

# Directory to save event-annotated frames
event_dir = 'annotated_event_frames'
os.makedirs(event_dir, exist_ok=True)

event_count = 0
timestamps = []

# FPS (adjust based on your video)
fps = 30  # Example FPS value

for i, motion_value in enumerate(motion_estimation):
    if motion_value > event_threshold:
        event_count += 1
        
        # Load the frame where the event occurs
        frame_path = os.path.join(frame_dir, frame_files[i + 1])  # +1 because motion is calculated for the next frame
        frame = cv2.imread(frame_path)
        
       
