import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

from delve.kerascallback import CustomTensorBoard, SaturationLogger

import tensorflow as tf


# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# Build model
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# Delve-specific
tbCallBack = CustomTensorBoard(log_dir='./runs', user_defined_freq=1)
saturation_logger = SaturationLogger(model, x_train[:2], print_freq=1)

# Train and evaluate model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Model(model.get_input_at(0), outputs=model.output)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# # Optional - save to csv
# csv_logger = keras.callbacks.CSVLogger('1.log')
model.fit(x_train, y_train,
          epochs=100,
          batch_size=128,
          callbacks=[saturation_logger])

score = model.evaluate(x_test, y_test, batch_size=128)

# Call `tensorboard --logdir=runs` to see layer saturation
