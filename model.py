import numpy as np
import tensorflow as tf
import csv
import cv2

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Cropping2D



lines = []
images = []
measurements = []

with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)



for line in lines:
    source_path = line[0]
    image = cv2.imread(source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    #add left camera image
    image_left = cv2.imread(line[1])
    measurements.append(measurement)

    # add right camera image
    image_right = cv2.imread(line[2])
    measurements.append(measurement)
    #Flip images
    if measurement != 0:
        image_flipped = np.fliplr(image)
        measurement_flipped = - measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)
        #Left camera images
        image_left_flipped = np.fliplr(image_left)
        images.append(image_left_flipped)
        measurements.append(measurement_flipped)
        #right camera images
        image_right_flipped = np.fliplr(image_right)
        images.append(image_left_flipped)
        measurements.append(measurement_flipped)



X_train = np.array(images)
Y_train = np.array(measurements)


def main(_):
    print("main function started")
    input_shape = (160, 320, 3)
    model = Sequential()
    #model.add(Cropping2D(cropping=((50, 10), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
    #model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=input_shape))
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
    model.save('model.h5')
    print("Saved model to disk")



if __name__ == '__main__':
    tf.app.run()
