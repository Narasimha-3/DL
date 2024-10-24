import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def CNN_model(X_train, Y_train, X_test, Y_test):
    # Define the model
    model = Sequential()

    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the layers
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with time estimation
    epochs = 10
    batch_size = 32
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, Y_test))
        epoch_end_time = time.time()
        
        elapsed_time = epoch_end_time - start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (epochs - epoch - 1)
        
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")
        print(f"Estimated remaining time: {remaining_time:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds.")