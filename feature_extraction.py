#Feature extraction

import numpy as np
import os
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model

# Check if GPU is available and being used
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU is available and being used.")
else:
    print("GPU is not available. Check your environment.")

# Frame extraction settings
IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
BATCH_SIZE = 32  # Process frames in batches for efficient GPU use

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Extract frames from video
def extract_frames_with_face_crop(video_path, num_frames=20):
    frames = []
    video_reader = cv2.VideoCapture(video_path)
    
    if not video_reader.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return frames
    
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / num_frames), 1)
    
    for frame_counter in range(num_frames):
        frame_pos = frame_counter * skip_frames_window
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        success, frame = video_reader.read()
        if not success:
            print(f"Warning: Frame at position {frame_pos} could not be read.")
            continue
        
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            # Get the largest detected face
            (x, y, w, h) = max(faces, key=lambda b: (b[2] * b[3]))
            face_frame = frame[y:y+h, x:x+w]  # Crop the face from the frame
            resized_frame = cv2.resize(face_frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        else:
            # If no face is detected, resize the whole frame
            resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Normalize the frame
        normalized_frame = resized_frame / 255.0
        frames.append(normalized_frame)
    
    video_reader.release()
    return np.array(frames)

# Load Xception model for feature extraction

def load_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    
    # Apply Global Average Pooling after the base model's output
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    model.trainable = False  # Freeze base model
    return model

# Extract features using Xception model in batches for GPU efficiency
def extract_features(frames, xception_model, batch_size=BATCH_SIZE):
    return xception_model.predict(frames, batch_size=batch_size)

# Process videos and save features
def process_videos_and_save_features(video_paths, labels, num_frames=100, feature_file='features_24gb.npz'):
    xception_model = load_xception_model()
    features = []
    flabel = []
    
    for idx, (video_path, label) in enumerate(zip(video_paths, labels)):
        print(f"Processing video {idx + 1}/{len(video_paths)}: {video_path}")
        
        # Extract frames with face cropping
        frames = extract_frames_with_face_crop(video_path, num_frames=num_frames)
        
        if len(frames) == num_frames:
            # Extract features using Xception (in batches to utilize GPU)
            feature = extract_features(frames, xception_model)
            features.append(feature)
            flabel.append(label)
        else:
            print(f"Skipped video {video_path} due to insufficient frames.")
        
        print(f"Processed video {idx + 1}/{len(video_paths)}")
    
    # Convert features list to a numpy array and save
    features = np.array(features)
    flabel = np.array(flabel)
    np.savez(feature_file, features=features, labels=flabel)
    print(f"Features saved to {feature_file}")

# Main function to process videos and save features
def main():
    csv_file = 'video_labels_final.csv'
    file_address = pd.read_csv(csv_file)
    video_paths = file_address.iloc[:, 0].values
    labels = file_address.iloc[:, 1].values
    
    process_videos_and_save_features(video_paths, labels, num_frames=10, feature_file='features_final.npz')

if __name__ == "__main__":
    main()
