import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
# load model
model = load_model('model33.h5')

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('14.jpeg')
filename = 'prediction.jpg'
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = cascade.detectMultiScale(img, 1.05, 5)

for x,y,w,h in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=4)
    roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    predictions = model.predict(img_pixels)

    # find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]

    cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

resized_img = cv2.resize(img, (500, 500))
cv2.imwrite(filename, resized_img)
cv2.imshow('Facial emotion analysis ', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
