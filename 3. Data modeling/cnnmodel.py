from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import pickle
import cv2
import numpy as np
import pandas as pd
import glob


def transform_image(image):
    image = image[int(1080 * 0.4):1080, :]
    image = cv2.resize(image, (200, 66))
    return image


def collect_data(dirname, csvname):
    with open('./sample.dat', 'ab') as f:
        data = []
        csv = pd.read_csv(csvname, sep=',')
        joy_values = csv['wheel'].values.tolist()

        images = glob.glob(dirname)

        count = 0

        for img in images:

            screenshot = cv2.imread(img)

            if count < len(joy_values):
                screenshot = transform_image(np.array(screenshot))

                data.append([screenshot, joy_values[count]])

            if count == len(images) - 1:
                print('Collected data count - {0}.'.format(count))
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                data = []  # clear the data from memory

            count += 1


def read_data(split=False):
    with open('./sample.dat', 'rb') as f:
        data = []
        while True:
            try:
                temp = pickle.load(f)

                if type(temp) is not list:
                    temp = np.ndarray.tolist(temp)

                data = data + temp
            except EOFError:
                break
        if split:
            x_train = []
            y_train = []

            for i in range(0, len(data)):
                x_train.append(data[i][0])
                y_train.append(data[i][1])

            return np.array(x_train), np.array(y_train)
        else:
            return np.array(data)


def create_model():
    nrows = 66
    ncols = 200
    img_channels = 3  # color channels
    output_size = 1

    model = Sequential()
    model.add(Dropout(0.35, input_shape=(nrows, ncols, img_channels)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu'))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_size))
    model.summary()

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_model():
    if os.path.isfile('./my_model.h5'):
        model = keras.models.load_model('my_model.h5')
    else:
        model = create_model()

    return model


def train_model(model):
    x, y = read_data(True)

    # test_size 0.15 : training data set 85%
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.15, random_state=1)

    model = get_model()

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=1, mode='auto')

    model.fit(x_train, y_train, epochs=8, batch_size=64, verbose=1,
              validation_data=(x_valid, y_valid), callbacks=[checkpoint])

    # save the model after training
    model.save('./my_model.h5')


def display_data():
    data = read_data()
    data_size = len(data)
    print('Data size  -  ' + str(data_size))

    for i in range(data_size - 1, 0, -1):
        image = data[i]
        cv2.imshow('window', image[0])
        print(str(image[1]))
        cv2.waitKey(50)
