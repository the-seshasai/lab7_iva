import os
import pickle

# Define paths for extracted features and output gender prediction
features_path = "extracted_features"
output_gender_path = "gender_predictions"

if not os.path.exists(output_gender_path):
    os.makedirs(output_gender_path)

def classify_gender(geo_features, texture_features):
    # Example rule-based classification
    if geo_features["jaw_width"] > 0 and geo_features["eye_distance"] < 0:
        return "Male"
    else:
        return "Female"

# Loop through extracted features
for filename in os.listdir(features_path):
    # Make sure to only process the feature files
    if filename.endswith(".pkl"):
        # Load the features saved with pickle
        with open(os.path.join(features_path, filename), 'rb') as f:
            features = pickle.load(f)
        
        geo_features = features['geo_features']
        texture_features = features['texture_features']
        
        # Classify gender based on the extracted features
        gender = classify_gender(geo_features, texture_features)
        
        # Save the result in a text file
        result_filename = os.path.join(output_gender_path, filename.replace(".pkl", "_gender.txt"))
        with open(result_filename, "w") as f:
            f.write(f"Identified Gender: {gender}\n")
    
print("Gender identification complete. Results saved in:", output_gender_path)
