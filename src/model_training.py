import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths for features CSV and saving model
FEATURES_CSV_PATH = "/Users/vanshtaneja/Desktop/emotion_detector/data/features/audio_features.csv"
MODEL_SAVE_PATH = "/Users/vanshtaneja/Desktop/emotion_detector/models/saved_models/audio_emotion_model.pkl"

def train_model():
    # Load the extracted features from CSV
    if not os.path.exists(FEATURES_CSV_PATH):
        print(f"Error: Features CSV file '{FEATURES_CSV_PATH}' not found.")
        return

    try:
        # Load features CSV into DataFrame
        features_df = pd.read_csv(FEATURES_CSV_PATH)
        print(f"Loaded features CSV: {FEATURES_CSV_PATH}")
        print(f"Number of rows in features DataFrame: {len(features_df)}")
        print("First few rows of the DataFrame:")
        print(features_df.head())

        # Split features and labels
        X = features_df.drop(['file_name', 'actor'], axis=1).values
        y = features_df['actor'].values
        print(f"Shape of features (X): {X.shape}")
        print(f"Shape of labels (y): {y.shape}")

        if X.shape[0] == 0 or y.shape[0] == 0:
            print("Error: Features or labels are empty after splitting. Please check the data.")
            return

    except KeyError as e:
        print(f"KeyError during feature-label split: {e}")
        return

    # Split the data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")
    except ValueError as e:
        print(f"ValueError during train-test split: {e}")
        return

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model for future use
    try:
        # Make sure the directory exists
        model_directory = os.path.dirname(MODEL_SAVE_PATH)
        if not os.path.exists(model_directory):
            print(f"Model directory '{model_directory}' does not exist. Creating it.")
            os.makedirs(model_directory)

        # Save the model using joblib
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"Model successfully saved to {MODEL_SAVE_PATH}")

    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    train_model()
