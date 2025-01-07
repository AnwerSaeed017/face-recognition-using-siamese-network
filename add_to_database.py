import numpy as np
import cv2
import os
import math
import warnings


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def AddToDatabase(image, BGR_input=True, BGR_output=True):
    """
    return GRAY cropped_face, x, y, w, h
    """
    gray = image.copy()
    if BGR_input:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=0,
        minSize=(10, 10)
    )
    if BGR_output:
        for (x, y, w, h) in faces:
            return image[y:y+h, x:x+w, :], x, y, w, h
    else:
        for (x, y, w, h) in faces:
            return gray[y:y+h, x:x+w], x, y, w, h
        


def resize_image(image, new_shape):
    # Resize the image to the new shape
    resized_image = cv2.resize(image, (new_shape[1], new_shape[0]))

    # If the image has more than one channel (e.g., RGB), convert it to grayscale
    if len(resized_image.shape) > 2:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

    # Add a new axis to make it 3D with one channel
    resized_image = resized_image[..., np.newaxis]

    return resized_image


def main():
    """
    
    database_folder = 'database/database'
    database_images = []
    database_names = []
    for file_name in os.listdir(database_folder):
        database_image = cv2.imread(os.path.join(database_folder, file_name), 0)
        database_images.append(database_image)
        database_names.append(os.path.splitext(file_name)[0])

    for img in database_images:
        print(f"shape: {img.shape}")
    
"""
    
    
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return

    # Create a window

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow("Webcam Press C to capture and Q to exit", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        # Check if 'c' key is pressed (for capture)
        if key == ord('c'):
            # Save the captured frame as an image
            face,x,y,w,h = AddToDatabase(frame)
            face = cv2.resize(face, (100, 100), interpolation=cv2.INTER_AREA)
           


            
            
            cv2.imwrite("database/database/Ahmed.jpg", face)
     
            print("Image captured!")

        # Check if 'q' key is pressed (for quit)
        elif key == ord('q'):
            break

    # Release the webcam
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    """

    input_image = cv2.imread("myself.jpg")


    face,x,y,w,h = AddToDatabase(input_image)
    face_resized = cv2.resize(face, (100, 100), interpolation=cv2.INTER_AREA)
    # Display the input image with bounding boxes
    cv2.imshow('Face Recognition', face_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
if __name__ == "__main__":
    main()
