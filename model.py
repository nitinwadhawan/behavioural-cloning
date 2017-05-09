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
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        #Left image
        # left_source_path = line[1]
        # left_filename = left_source_path.split('/')[-1]
        # left_current_path = 'data/IMG/' + filename
        # left_image = cv2.imread(left_current_path)
        # images.append(left_image)
        # left_measurement = float(line[3]) + 0.25
        # measurements.append(left_measurement)
        # # Right image
        # right_source_path = line[2]
        # right_filename = right_source_path.split('/')[-1]
        # right_current_path = 'data/IMG/' + filename
        # right_image = cv2.imread(right_current_path)
        # images.append(right_image)
        # right_measurement = float(line[3]) - 0.25
        # measurements.append(right_measurement)

        if measurement != 0:
            image_flipped = np.fliplr(image)
            measurement_flipped = - measurement
            images.append(image_flipped)
            measurements.append(measurement_flipped)
            # right_flipped = np.fliplr(right_image)
            # images.append(right_flipped)
            # measurements.append(measurement_flipped)
            # left_flipped = np.fliplr(left_image)
            # images.append(left_flipped)
            # measurements.append(measurement_flipped)


    # correction = 0.2  # this is a parameter to tune
    # steering_left = measurement + correction
    # steering_right = measurement - correction
    #add left camera image
    # left_source_path = line[1]
    # image_left = cv2.imread(left_source_path)
    # left_meas = (measurement+0.25)

X_train = np.array(images)
Y_train = np.array(measurements)


def main(_):
    print("main function started")
    input_shape = (160, 320, 3)
    model = Sequential()
    #model.add(Cropping2D(cropping=((50, 10), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
    #model.add(Cropping2D(cropping=((50, 10), (0, 0)), input_shape=input_shape))
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
    model.fit(X_train, Y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
    model.save('model.h5')
    print("Saved model to disk")



if __name__ == '__main__':
    tf.app.run()
