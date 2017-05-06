import matplotlib.image as mpimg
import numpy as np
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math

colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('driving_log.csv', skiprows=[0], names=colnames)
center_images = data.center.tolist()
left = data.left.tolist()
right = data.right.tolist()
measurements = data.steering.tolist()
throttle = data.throttle.tolist()

# center_images, measurements = shuffle(center_images, measurements)
# center_images, X_valid, measurements, y_valid = train_test_split(center_images, measurements, test_size=0.10, random_state=100)
images = []
meas = []


for measurement in measurements:
    meas.append(float(measurement))

for im in center_images:
    images.append(im)

X_train = np.array(images)
Y_train = np.array(meas)


def main(_):
    print("main function started")
    # valid_generator = valid_generator(X_valid, y_valid, 128)
    print("data generated")
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(80, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(40, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(16, W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1, W_regularizer=l2(0.001)))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.summary()


    ### Model training
    model.fit(X_train, Y_train,validation_split=0.2,shuffle=True)

    # model_json =  model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("model.h5")
    model.save('model.h5')
    print("Sa ved model to disk")


if __name__ == '__main__':
    tf.app.run()
