import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

FEATURES_CSV_PATH = "/Users/vanshtaneja/Desktop/emotion_detector/data/features/audio_features.csv"
MODEL_PATH = "/Users/vanshtaneja/Desktop/emotion_detector/models/saved_models/audio_emotion_model.pkl"

def evaluate_model():
    # Load the extracted features from CSV
    if not os.path.exists(FEATURES_CSV_PATH):
        print(f"Error: Features CSV file '{FEATURES_CSV_PATH}' not found.")
        return

    # Load features and labels
    features_df = pd.read_csv(FEATURES_CSV_PATH)
    X = features_df.drop(['file_name', 'actor'], axis=1).values
    y = features_df['actor'].values

    # Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    model = joblib.load(MODEL_PATH)

    # Predict on the entire dataset (or you can use a split dataset if needed)
    y_pred = model.predict(X)

    # Evaluate the model's performance
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
