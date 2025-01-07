import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# Define the Siamese network architecture
def create_siamese_net():
    input_shape = (100, 100, 1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    negative_input = Input(input_shape)

    # Define the shared layers
    convnet = tf.keras.Sequential([
        Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(2e-4)),
        Flatten(),
        Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3))
    ])

    # Calculate the output with a lambda layer for triplet loss
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    encoded_n = convnet(negative_input)
    loss = Lambda(triplet_loss)([encoded_l, encoded_r, encoded_n])
    
    model = Model(inputs=[left_input, right_input, negative_input], outputs=loss)
    model.compile(loss='mean_squared_error', optimizer=Adam(0.00006))
    return model

def triplet_loss(x, alpha=0.2):
    anchor, positive, negative = x
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + alpha
    return K.maximum(basic_loss, 0.0)

# Function to load weights into the model
def load_network(weights_path):
    model = create_siamese_net()
    try:
        model.load_weights(weights_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading the weights: {e}")
        return None
    return model


# Haarcascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    return faces

# Function to compare the webcam face against the database
def compare_with_database(frame, model, database_folder="database/database"):
    if model is None:
        print("Model not loaded, skipping comparison.")
        return
    # Load images as grayscale
    database_images = [cv2.imread(os.path.join(database_folder, f), cv2.IMREAD_GRAYSCALE) for f in os.listdir(database_folder)]
    database_names = [os.path.splitext(f)[0] for f in os.listdir(database_folder)]

    faces = detect_faces(frame)
    for (x, y, w, h) in faces:
        face = cv2.resize(frame[y:y+h, x:x+w], (100, 100))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        predictions = model.predict([np.array([face]) for _ in range(3)])  # Provide three identical inputs
        match_index = np.argmax(predictions)
        match_score = predictions[match_index]

        person_name = "Unknown"
        if match_score > 0.5:
            person_name = database_names[match_index]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name}: {match_score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def main():
    model = load_network('saved_best (1).h5')
    if model is None:
        print("Model could not be loaded. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        compare_with_database(frame, model)
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
