# Prediction

def extract_features_for_prediction(video_path, num_frames=10):
    """
    Extracts features from the video for prediction by first extracting and 
    preprocessing frames, then using the Xception model to get feature vectors.
    """
    # Load the Xception model for feature extraction
    xception_model = load_xception_model()

    # Extract frames with face cropping and resizing
    frames = extract_frames_with_face_crop(video_path, num_frames=num_frames)
    
    # Ensure we have the correct number of frames
    if len(frames) < num_frames:
        print(f"Warning: Only {len(frames)} frames extracted; required {num_frames}.")
        return None
    
    # Extract features
    features = xception_model.predict(frames)
    return np.array(features)


# model_training.py (Additions for Prediction)

# from feature_extraction import extract_features_for_prediction, load_xception_model  # Import feature extraction methods

def load_trained_model(model_path='trained_lstm_model.h5'):
    """
    Load the trained LSTM model.
    """
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Trained model file not found at {model_path}")

def predict_video(video_path, num_frames=10, model_path='trained_lstm_model.h5'):
    """
    Predict whether a video is deepfake or real by extracting features and
    using the trained LSTM model to classify.
    """
    # Load the trained LSTM model
    model = load_trained_model(model_path)
    
    # Extract features from the video for prediction
    features = extract_features_for_prediction(video_path, num_frames=num_frames)
    
    # If feature extraction failed, return None
    if features is None:
        print("Failed to extract features for prediction.")
        return None

    # Reshape features to match model input (batch size of 1)
    features = features.reshape(1, num_frames, -1)
    
    # Predict with the model
    prediction = model.predict(features)
    print(prediction)
    # Interpret prediction result
    is_deepfake = prediction[0][0] > 0.5  # Assumes threshold of 0.5 for binary classification
    return "Deepfake" if is_deepfake else "Real"

# Example usage of predict_video function

video_path = r"D:\FF\manipulated_sequences\Deepfakes\c23\videos\000_003.mp4"  # Replace with your video path
prediction = predict_video(video_path, num_frames=10, model_path='final_mix.keras')
print(f"The video is predicted to be: {prediction}")

