import os
import pandas as pd
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix


from skimage.transform import resize
from skimage.io import imread

import pickle

import os
import imageio

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# capture frames from a camera
new_image_size = (224,224,3)





img = cv2.imread('eye2.jpeg')

model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()



# Detects faces of different sizes in the input image
faces = eye_cascade.detectMultiScale(img, 1.3, 5)

print(faces)
i=0
for (x, y, w, h) in faces:

    bee = img[y:y+h, x:x+w]
    zoo='kar'+str(i)+'.jpg'
    i+=1

    #cv2.imwrite( zoo,bee)
    # To draw a rectangle in a face
    cv2.rectangle(img, (x, y), (x + w, y + h), (250,250,2), 2)
    roi_gray = img[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    img_resize = resize(roi_gray, new_image_size)
    cv2.imwrite("main.jpg",img_resize)


    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("main.jpg").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)



# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model


# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1

