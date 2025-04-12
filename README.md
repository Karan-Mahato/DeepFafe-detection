# **Deep-Fake Detection Model**

## **Overview**
This project is a deepfake detection system that combines **XceptionNet** for feature extraction and **Bidirectional LSTM** for video classification. The model processes video frames, extracts deep visual features, and determines whether a video is real or manipulated.

## **Dataset**
The dataset used consists of deepfake and real videos from multiple sources:
- **Celeb-DF**
- **FaceForensics++**
- **DFDC (DeepFake Detection Challenge)**

## **Project Structure**
```
Deepfake_Detection/
│── feature_extraction.py   # Extracts features from videos using Xception
│── model_training.py       # Trains the LSTM model for classification
│── prediction.py           # Loads model and predicts deepfake videos
│── requirements.txt        # Required Python libraries
│── README.md               # Project documentation
```

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/deepfake-detection.git
cd deepfake-detection
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Ensure GPU is Available**
Check if TensorFlow detects GPU acceleration:
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

## **Feature Extraction**
Extract features from videos using **XceptionNet**:
```python
python feature_extraction.py
```
This will:
- Extract faces from video frames
- Resize & normalize images
- Use Xception to extract features
- Save features as a `.npz` file

## **Training the LSTM Model**
Train the **Bidirectional LSTM** classifier:
```python
python model_training.py
```
- Uses extracted features to train a deepfake classifier
- Implements **EarlyStopping** to prevent overfitting
- Saves the trained model as `final_mix.keras`

## **Prediction**
Predict whether a given video is a deepfake:
```python
python prediction.py --video_path "path/to/video.mp4"
```
This will:
- Extract video frames
- Process them through the Xception model
- Feed extracted features into the trained LSTM model
- Print whether the video is **Real** or **Deepfake**

## **Evaluation Metrics**
The model is evaluated using:
- **Accuracy**
- **Loss (Binary Crossentropy)**
- **Precision & Recall**
- **Confusion Matrix**

To plot training performance:
```python
python plot_training.py
```

## **Results & Performance**
- **Achieved over 94% accuracy** on validation datasets.
- Robust detection of various deepfake techniques.
- Handles real-world videos with varying resolutions & lighting conditions.
- ![image](https://github.com/user-attachments/assets/4fdee243-8639-4b8f-a9e3-20c654b3f412)


## **Future Improvements**
- Implement **attention mechanisms** for improved feature selection.
- Integrate **temporal coherence analysis** to enhance deepfake detection.
- Explore **GAN-based adversarial training** for robustness.



