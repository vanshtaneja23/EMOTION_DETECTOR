import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import joblib
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import tempfile
import pyaudio
import wave

# Constants
MODEL_PATH = "/Users/vanshtaneja/Desktop/final_emotion_detector/audio_emotion_model.pkl"
# Load the model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("Model file not found.")
        return None

model = load_model()

# Function to extract features from audio
def extract_features_from_audio(audio, sample_rate=16000):
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
    features = np.concatenate((mfccs, chroma, spectral_contrast)).reshape(1, -1)
    return features

# Streamlit App
st.title("Real-Time Audio Emotion Detector 🎙️")
st.markdown("### Record an audio sample and detect its emotion")

# Explanation of the Model
st.markdown("## How the Emotion Detection Model Works")
st.write("The emotion detection model is built using machine learning techniques. Here's a brief overview of how it works:")

st.markdown("### 1. Audio Feature Extraction")
st.write("The model takes an audio sample as input and extracts key features from it. These features include:")
st.write("- **MFCCs (Mel Frequency Cepstral Coefficients)**: These are coefficients that represent the short-term power spectrum of sound and are commonly used in audio processing.")
st.write("- **Chroma Feature**: This feature relates to the twelve different pitch classes, providing information about the harmonic content of the audio.")
st.write("- **Spectral Contrast**: This captures the difference in amplitude between peaks and valleys in a sound spectrum, offering insight into the timbral texture of the audio.")

st.markdown("### 2. Model Training")
st.write("The extracted features are used to train a machine learning model. In this case, we have used a Random Forest Classifier, which is trained on a labeled dataset containing various emotional states like happy, sad, angry, etc.")

st.markdown("### 3. Emotion Prediction")
st.write("After training, the model is able to classify new audio inputs based on the extracted features, predicting the emotion that best matches the characteristics of the input.")

# Record Audio using sounddevice
duration = st.slider("Select Recording Duration (seconds):", 1, 10, 5)
record_button = st.button("Record Audio with Sounddevice 🎤")

if record_button:
    st.info("Recording in progress... Speak now!")
    audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1)
    sd.wait()  # Wait until the recording is done
    audio_data = audio_data.flatten()

    # Create a temporary file to save the recording
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        write(tmp_file.name, 16000, audio_data)
        temp_audio_file_path = tmp_file.name

    st.success("Recording complete!")
    
    # Plot waveform of the recorded audio
    st.markdown("### Recorded Audio Waveform")
    fig, ax = plt.subplots()
    ax.plot(audio_data)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Predict Emotion
    if model:
        st.markdown("### Predicting Emotion...")
        features = extract_features_from_audio(audio_data)
        predicted_emotion = model.predict(features)[0]
        st.write(f"**Predicted Emotion:** {predicted_emotion}")
    else:
        st.error("Model could not be loaded.")

# Record Audio using pyaudio
duration_pyaudio = st.slider("Select Recording Duration for PyAudio (seconds):", 1, 10, 5)
record_pyaudio_button = st.button("Record Audio with PyAudio 🎙️")

if record_pyaudio_button:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    st.info("Recording in progress... Speak now!")
    for _ in range(0, int(RATE / CHUNK * duration_pyaudio)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recording to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        wf = wave.open(tmp_file.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        temp_audio_file_path_pyaudio = tmp_file.name

    st.success("Recording complete with PyAudio!")

    # Plot waveform of the recorded audio
    st.markdown("### Recorded Audio Waveform with PyAudio")
    fig, ax = plt.subplots()
    audio_data_pyaudio, _ = librosa.load(temp_audio_file_path_pyaudio, sr=RATE)
    ax.plot(audio_data_pyaudio)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Predict Emotion
    if model:
        st.markdown("### Predicting Emotion...")
        features = extract_features_from_audio(audio_data_pyaudio)
        predicted_emotion = model.predict(features)[0]
        st.write(f"**Predicted Emotion:** {predicted_emotion}")
    else:
        st.error("Model could not be loaded.")

# File Upload Option
st.markdown("### Upload an Audio File for Emotion Detection")
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        uploaded_audio_path = tmp_file.name

    # Load audio and display waveform
    audio, sr = librosa.load(uploaded_audio_path, sr=16000)
    st.markdown("### Uploaded Audio Waveform")
    fig, ax = plt.subplots()
    ax.plot(audio)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Predict Emotion
    if model:
        st.markdown("### Predicting Emotion...")
        features = extract_features_from_audio(audio)
        predicted_emotion = model.predict(features)[0]
        st.write(f"**Predicted Emotion:** {predicted_emotion}")
    else:
        st.error("Model could not be loaded.")

st.markdown("---")
st.markdown("**Vansh Taneja**")
