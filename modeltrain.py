from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

(a_train, b_train), (a_test, b_test) = cifar10.load_data()

a_train = a_train.astype('float32') / 255.0
a_test = a_test.astype('float32') / 255.0

b_train = to_categorical(b_train, 10)
b_test = to_categorical(b_test, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(a_train, b_train, batch_size=64, epochs=10, validation_data=(a_test, b_test))

model.save('my_model.h5')