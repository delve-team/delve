from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import time, os
import numpy as np
from delve.kerascallback import CustomTensorBoard, SaturationLogger

# tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# import ipdb; ipbd.set_trace()


input_num = np.linspace(-100, 100, 50000)
np.append(np.linspace(-20, 20, 10000) + 0.3, input_num)
square = np.square(input_num)

validation_data = input_num + 0.5
validation_squared = np.square(validation_data)

for i, c in enumerate(input_num):
    print("{} input = {} squared".format(c, square[i]))

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(units=512, input_shape=[1], activation='relu'),
     tf.keras.layers.Dropout(0.4),
     tf.keras.layers.Dense(units=1024, activation='relu'),
     tf.keras.layers.Dense(units=8, activation='relu'),
     tf.keras.layers.Dense(units=1, activation='linear')
     ])

saturation_logger = SaturationLogger(model, input_data=input_num[:10], print_freq=1)

start = time.time()

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0005))

history = model.fit(input_num, square, epochs=50, verbose=True, batch_size=64, callbacks=[saturation_logger])
print("Finished training the model")

score = model.evaluate(validation_data, validation_squared, batch_size=64)
print(score)

end = (time.time() - start)
print("\nTime taken: ", (end / 60), "mins")

print(model.summary())

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
print("\nModel", model.history)
plt.show()

# user_number = (input('Enter number:'))

for i in range(-10, 10, 1):
    prediction = model.predict([i])
    print("\nSquare: ", i, int(prediction))

# tf.keras.models.save_model(model, "./models/model.square_sgd", overwrite=True, include_optimizer=True)

# print("These are the layer variables: {}".format(l0.get_weights()))
# print("These are the layer variables: {}".format(l1.get_weights()))
# print("These are the layer variables: {}".format(l2.get_weights()))
