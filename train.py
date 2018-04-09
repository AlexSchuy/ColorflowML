import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization, Dense, Flatten
from tensorflow.python.keras.regularizers import l2
import os

import data
import constants

batch_size = 96
epochs = 100
save_dir = os.path.join(constants.data_dir, 'saved_models')
model_name = 'keras_colorflow_trained_model.h5'

# Load the pixel data, split it into train/test and standardize it.
X_octet = data.get_pixels(octet=True)
X_singlet = data.get_pixels(octet=False)
y_octet = np.zeros(X_octet.shape[0])
y_singlet = np.ones(X_singlet.shape[0])
X = np.concatenate([X_octet, X_singlet])
y = np.concatenate([y_octet, y_singlet])
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("[train] Standardizing data ...")
scaler = StandardScaler()
X_train = np.reshape(scaler.fit_transform(X_train), (X_train.shape[0], 25, 25, 1))
X_test = np.reshape(scaler.transform(X_test), (X_test.shape[0], 25, 25, 1))
print("[train] training data shape: {}".format(X_train.shape))
print("[train] testing data shape: {}".format(X_test.shape))

model = Sequential()

# Convolution 1
model.add(Dropout(0.5, input_shape=(25, 25, 1)))
model.add(Conv2D(32, (11, 11), padding='same', activation='relu', kernel_regularizer=l2()))
model.add(MaxPooling2D((2, 2)))

# Convolution 2
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2()))
model.add(MaxPooling2D((3, 3)))

# Convolution 3
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2()))
model.add(MaxPooling2D((3, 3)))

model.add(BatchNormalization())

# Fully-Connected 1
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2()))

# Fully-Connected 2
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1, activation='relu', kernel_regularizer=l2()))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, y_test), shuffle=True)

if not os.path.isdir(save_dir):
  os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('saved trained model at %s ' % model_path)

scores = model_evaluate(X_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
