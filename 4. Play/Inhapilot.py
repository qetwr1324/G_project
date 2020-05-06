from PIL import ImageGrab
import numpy as np
import cv2
import time
import os
import keras


modelpath = './mymodel.h5'
def get_model():
    model = keras.models.load_model(modelpath)
    return model

def transform_image(image):
    image = image[int(1080 * 0.4):1080, :]
    image = cv2.resize(image, (200, 66))
    return image


if __name__ == "__main__":
    model = get_model()
    while(True):
        curTime = time.time()
        img = ImageGrab.grab(bbox=(0, 648, 1920, 1080))
        img = img.resize((200, 66))
        img=cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)
        wheel = model.predict(np.expand_dims(img,axis=0))
        print(wheel)


