import numpy as np
import cv2
import os
import math
import warnings
warnings.filterwarnings("ignore")

from model import create_siamese_net

# Instantiate the SiameseNet model using create_siamese_net function
siamese_net = create_siamese_net()
siamese_net.load_weights('saved_best.h5')

# Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def giveAllFaces(image, BGR_input=True, BGR_output=False):
    """
    return GRAY cropped_face, x, y, w, h
    """
    gray = image.copy()
    if BGR_input:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if BGR_output:
        for (x, y, w, h) in faces:
            yield image[y:y+h, x:x+w, :], x, y, w, h
    else:
        for (x, y, w, h) in faces:
            yield gray[y:y+h, x:x+w], x, y, w, h

def compare_with_database(input_image, database_folder="database/database"):
    # Load the database images
    database_images = []
    database_names = []
    for file_name in os.listdir(database_folder):
        database_image = cv2.imread(os.path.join(database_folder, file_name), 0)
        database_images.append(database_image)
        database_names.append(os.path.splitext(file_name)[0])

    for face, x, y, w, h in giveAllFaces(input_image):
        face_resized = cv2.resize(face, (100, 100), interpolation=cv2.INTER_AREA)
        face_resized = np.expand_dims(face_resized, -1)
        left = np.array([face_resized for _ in range(len(database_images))])
        right = np.array([np.expand_dims(image, -1) for image in database_images])
        probs = np.squeeze(siamese_net.predict([left, right]))
        index = np.argmax(probs)
        
        if probs.ndim > 0:
            prob = probs[index] if index < len(probs) else None
        else:
            prob = None

        name = "Unknown"
        if prob is not None and prob > 0.5:
            name = database_names[index]

        # Format the probability value
        prob_text = "{:.2f}".format(prob) if prob is not None else "N/A"

        putBoxText(input_image, x, y, w, h, text=name + "({})".format(prob_text))

def putBoxText(image, x, y, w, h, text="unknown"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, text, (x, y - 6), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

if __name__ == "__main__":
    input_image = cv2.imread(r"C:\Users\Administrator\Downloads\myself (2).jpg")  # Update path to your input image
    compare_with_database(input_image)
    cv2.imshow('Face Recognition', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
