import os
import librosa
import numpy as np
import pandas as pd

# Update path to the correct directory structure
PROCESSED_DATA_PATH = "data/processed/"
FEATURES_PATH = "data/features/"

def extract_features():
    """
    Extract features like MFCCs, Chroma, and Spectral Contrast from the audio files
    and save them in a CSV format for model training.
    """
    # Create features directory if it doesn't exist
    if not os.path.exists(FEATURES_PATH):
        os.makedirs(FEATURES_PATH)

    features_list = []

    # Debugging print to ensure directory exists
    print(f"Processing directory: {PROCESSED_DATA_PATH}")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data path '{PROCESSED_DATA_PATH}' does not exist.")
        return

    # Iterate through all processed audio files
    for root, dirs, files in os.walk(PROCESSED_DATA_PATH):
        print(f"Accessing directory: {root}, Number of files: {len(files)}")
        if len(files) == 0:
            print("Warning: No files found in this directory.")
        
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    # Load the audio file
                    print(f"Attempting to load: {file_path}")
                    audio, sample_rate = librosa.load(file_path, sr=None)
                    print(f"Successfully loaded {file_path}")

                    # Extract MFCCs (Mel Frequency Cepstral Coefficients)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                    mfccs = np.mean(mfccs.T, axis=0)
                    print(f"MFCCs extracted for {file_path}")

                    # Extract Chroma features
                    stft = np.abs(librosa.stft(audio))
                    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
                    chroma = np.mean(chroma.T, axis=0)
                    print(f"Chroma features extracted for {file_path}")

                    # Extract Spectral Contrast
                    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
                    spectral_contrast = np.mean(spectral_contrast.T, axis=0)
                    print(f"Spectral Contrast extracted for {file_path}")

                    # Extract the actor label from folder name
                    actor_label = os.path.basename(root)
                    
                    # Combine all features into a single array
                    features = np.concatenate((mfccs, chroma, spectral_contrast))
                    
                    # Append to the feature list along with the label
                    features_list.append([file, actor_label] + features.tolist())

                    print(f"Extracted features from {file_path}")
                
                except Exception as e:
                    print(f"Error extracting features from {file_path}: {e}")

    # Debugging print statement to ensure features were extracted
    if len(features_list) == 0:
        print("No features were extracted. Please check the audio files and preprocessing steps.")
    else:
        print(f"Extracted features for {len(features_list)} files.")

    # Convert the feature list to a DataFrame
    columns = ['file_name', 'actor'] + [f'mfcc_{i}' for i in range(13)] + \
              [f'chroma_{i}' for i in range(12)] + \
              [f'spectral_contrast_{i}' for i in range(7)]
    features_df = pd.DataFrame(features_list, columns=columns)
    
    
    
    # Save features to a CSV file
    features_csv_path = os.path.join(FEATURES_PATH, "audio_features.csv")
    features_df.to_csv(features_csv_path, index=False)
    print(f"Features saved to {features_csv_path}")

if __name__ == "__main__":
    extract_features()
