import joblib
import librosa
import numpy as np
import os

MODEL_PATH = "/Users/vanshtaneja/Desktop/emotion_detector/models/saved_models/audio_emotion_model.pkl"

def predict_emotion(audio_file_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' not found.")
        return

    model = joblib.load(MODEL_PATH)

    try:
        # Load and extract features from the audio file
        audio, sample_rate = librosa.load(audio_file_path, sr=16000)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)

        # Combine features
        features = np.concatenate((mfccs, chroma, spectral_contrast)).reshape(1, -1)

        # Predict emotion
        predicted_emotion = model.predict(features)
        print(f"Predicted Emotion: {predicted_emotion[0]}")

    except Exception as e:
        print(f"Error while processing audio file '{audio_file_path}': {e}")

if __name__ == "__main__":
    # Example usage - replace with your actual audio file path
    predict_emotion("/Users/vanshtaneja/Desktop/emotion_detector/generated_audio.wav")

