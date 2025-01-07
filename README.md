# Siamese Network-Based Face Recognition

This project implements a face recognition system using a Siamese Neural Network and triplet loss to identify and compare faces. The system leverages OpenCV for face detection, TensorFlow for the deep learning model, and a pre-trained model for predictions.

## Features

- **Face Detection**: Uses Haarcascade to detect faces in an image or video stream.
- **Siamese Network**: Deep learning architecture to compare facial embeddings using triplet loss.
- **Real-Time Face Recognition**: Compare live webcam feed faces with a database of stored faces.
- **Database Matching**: Identify faces against a database of pre-processed images.
- **Custom Model Training**: Option to retrain or fine-tune the Siamese Network.

---

## Files

### 1. **Model Definition**
The Siamese network is defined in `create_siamese_net()`:
- Input shape: `(100, 100, 1)` for grayscale images.
- Shared layers include convolutional, pooling, and dense layers.
- The triplet loss function ensures embedding optimization.

### 2. **Face Detection**
Uses OpenCV's Haarcascade (`haarcascade_frontalface_default.xml`) to locate faces within the frame. The `detect_faces()` function returns detected face coordinates.

### 3. **Real-Time Webcam Integration**
The `main()` function uses a webcam feed to capture live frames, detect faces, and compare them with the database using the pre-trained Siamese Network.

### 4. **Database Matching**
The system compares detected faces with a database of pre-saved images (`database/database`) to find the best match based on model predictions.

---

## Requirements

Install the required dependencies using the following:

```bash
pip install tensorflow opencv-python numpy
```

---

## Usage

### 1. **Prepare the Database**
- Create a folder named `database/database`.
- Add images of people in grayscale (size 100x100) with their name as the filename.

### 2. **Run the Application**
To start the face recognition application:

```bash
python main.py
```

Press `q` to exit the webcam feed.

### 3. **Custom Image Comparison**
Use the `compare_with_database()` function to compare a static image against the database.

---

## Siamese Network Architecture

- **Convolutional Layers**: Extract spatial features from images.
- **Fully Connected Layers**: Learn compact embeddings for faces.
- **Triplet Loss**: Optimizes the embeddings to minimize distance for the same identity and maximize distance for different identities.

```python
def triplet_loss(x, alpha=0.2):
    anchor, positive, negative = x
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + alpha
    return K.maximum(basic_loss, 0.0)
```

---

## Pre-Trained Model
The pre-trained model weights are loaded using:

```python
model.load_weights('saved_best.h5')
```
Replace the path with your custom-trained model weights if necessary.

---

## Future Enhancements

- **Data Augmentation**: Increase dataset variety using image augmentation techniques.
- **Multiclass Classification**: Expand recognition capabilities to multiple identities simultaneously.
- **GPU Optimization**: Speed up real-time predictions using GPU acceleration.

---


