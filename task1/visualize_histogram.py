import cv2
import os
import matplotlib.pyplot as plt

# Directory where grayscale frames are saved
frame_dir = 'frames'

# Get the list of frame filenames
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))

# Create a directory to save the histograms
histogram_dir = 'histograms'
os.makedirs(histogram_dir, exist_ok=True)

for frame_file in frame_files:
    # Load the grayscale frame
    gray_frame = cv2.imread(os.path.join(frame_dir, frame_file), cv2.IMREAD_GRAYSCALE)

    # Calculate histogram
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])

    # Plot and save the histogram
    plt.figure()
    plt.title(f"Histogram for {frame_file}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    
    hist_file = os.path.join(histogram_dir, f'hist_{frame_file}.png')
    plt.savefig(hist_file)
    plt.close()

    print(f"Histogram saved for {frame_file}")
