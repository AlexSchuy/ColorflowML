from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Dense
from keras.regularizers import l2
import os

import data

batch_size = 96
epochs = 100
save_dir = os.path.join(data_dir, 'saved_models')
model_name = 'keras_colorflow_trained_model.h5'

# Load the pixel data, split it into train/test and standardize it.
X_octet = data.get_pixels(octet=True)
X_singlet = data.get_pixels(octet=False)
y_octet = np.zeros(X_octet.shape[0])
y_singlet = np.ones(X_singlet.shape[0])
X = np.concat(X_octet, X_singlet)
y = np.concat(y_octet, y_singlet)
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

# Convolution 1
model.add(Dropout(0.5))
model.add(Conv2D(32, (11, 11), padding='same', activation='relu', kernel_regularizer=l2()))
model.add(MaxPool((2, 2)))

# Convolution 2
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2()))
model.add(MaxPool((3, 3)))

# Convolution 3
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2()))
model.add(MaxPool((3, 3)))

model.add(BatchNormalization())

# Fully-Connected 1
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2()))

# Fully-Connected 2
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu', kernel_regularizer=l2()))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_train, y_train), shuffle=True)

if not os.path.isdir(save_dir):
  os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('saved trained model at %s ' % model_path)

scores = model_evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
