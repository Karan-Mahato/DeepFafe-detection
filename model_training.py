import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load features from the saved .npz file and handle key errors
def load_features(feature_file='features_24gb_sonu.npz'):
    if os.path.exists(feature_file):
        print(f"Loading features from {feature_file}")
        with np.load(feature_file) as data:
            print(f"Available keys in the .npz file: {data.keys()}")  # List available keys
            key = list(data.keys())[0]  # Automatically select the first key or specify the correct key
            features = data[key]  # Access the correct array
        return features
    else:
        raise FileNotFoundError(f"{feature_file} does not exist. Please process the videos to extract features.")

# Build the LSTM-based classification model
def create_model(sequence_length, feature_size):
    video_input = Input(shape=(sequence_length, feature_size))
    
    # Use Bidirectional LSTM
    x = Bidirectional(LSTM(16, return_sequences=False))(video_input)
    x = Dropout(0.3)(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=video_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Function to plot the training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(np.array(history.history['accuracy']) * 100)  # Scale accuracy to 0-100
    plt.plot(np.array(history.history['val_accuracy']) * 100)  # Scale validation accuracy
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)  # Set y-axis limits between 0 and 100
    plt.legend(['Train', 'Validation'])
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)  # Set y-axis limits between 0 and 1 for loss
    plt.legend(['Train', 'Validation'])
    
    plt.show()

# Main function to load features and train the LSTM model
def main(labels, num_frames=10, feature_file='/kaggle/input/features/features.npz'):
    # Load pre-extracted features
    features = load_features(feature_file)
    
    # Convert labels to numpy array
    labels = np.array(labels)

    # Shuffle features and labels
    indices = np.arange(features.shape[0])  # Create an array of indices
    np.random.shuffle(indices)  # Shuffle the indices
    features = features[indices]  # Shuffle features
    labels = labels[indices]      # Shuffle labels
    
    # Create and compile the LSTM model
    model = create_model(sequence_length=num_frames, feature_size=features.shape[-1])
    
    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model using the pre-extracted features
    hist = model.fit(features, labels, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    print(hist)
    # Plot training history
    plot_training_history(hist)

    # Save the model as 'final_model'
    model.save('final_mix.keras')  # Save the model to a file (in HDF5 format)
    print("Model saved as final_mix.keras")

    return hist

# Load video labels from the CSV
def load_labels(csv_file='video_labels_sonu.csv'):
    file_address = pd.read_csv(csv_file)
    labels = file_address['label'].values
    return labels

if __name__ == "__main__":  # Corrected the if condition
    labels = load_labels('video_labels_final.csv')
    feature_file = 'features_final.npz'
    hist = main(labels, num_frames=10, feature_file=feature_file)
