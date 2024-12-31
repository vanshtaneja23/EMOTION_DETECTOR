import os
import librosa
import soundfile as sf

# Update the path to match the correct directory structure
RAW_DATA_PATH = "data/Emotions"
PROCESSED_DATA_PATH = "data/processed"

def preprocess_audio_files():
    """
    Preprocess audio files by loading each file, converting to a consistent format,
    and saving the processed file to the processed directory.
    """
    # Create processed directory if it doesn't exist
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    
    print(f"Starting preprocessing. Raw data path: {RAW_DATA_PATH}")

    # Iterate through all raw audio files in actor folders
    for root, _, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    print(f"Processing file: {file_path}")

                    # Load audio file and convert to 16kHz sample rate
                    audio, sample_rate = librosa.load(file_path, sr=16000)
                    print(f"Loaded file: {file_path}, Sample rate: {sample_rate}, Audio length: {len(audio)}")

                    # Check if the audio is empty or corrupted
                    if len(audio) == 0:
                        print(f"Warning: Audio file {file_path} is empty or corrupted. Skipping.")
                        continue

                    # Extract actor folder name
                    actor_folder = os.path.basename(root)

                    # Create processed folder for each actor
                    processed_folder_path = os.path.join(PROCESSED_DATA_PATH, actor_folder)
                    if not os.path.exists(processed_folder_path):
                        os.makedirs(processed_folder_path)

                    # Define path for processed audio file
                    processed_file_path = os.path.join(processed_folder_path, file)

                    # Save the processed audio file
                    sf.write(processed_file_path, audio, sample_rate)
                    
                    print(f"Processed and saved file to: {processed_file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Check if any files were saved
    if not os.path.exists(PROCESSED_DATA_PATH) or len(os.listdir(PROCESSED_DATA_PATH)) == 0:
        print(f"Error: No processed files found in {PROCESSED_DATA_PATH}. Please check preprocessing steps.")
    else:
        print(f"Preprocessing complete. Files saved in {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_audio_files()
