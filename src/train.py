import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import utils
from datetime import datetime

from tensorflow.keras.preprocessing.image import random_rotation, random_brightness

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping


def fit_mode(train, test, savedir, num_layers=2,
                  num_filters=128, kernel_size=3, pool_size=2, dropout_rate=0.2, learning_rate=0.001,
                  n_epochs=100, batch_size=96, verbose=1):

    model = Sequential()
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _ in range(num_layers):
        model.add(Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callback = EarlyStopping(monitor='loss', patience=3)

    history = model.fit(train, validation_data=test,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        callbacks=[callback])
    model.summary()

    print(history)

    scores = model.evaluate(test)
    print("Loss: %.2f%%" % (scores[0]))
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    model_path = os.path.join(savedir, 'saved_model_' + date_time + '.h5')  # Saving as HDF5 file
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model


def fit_cnn_model_sean(X_train, Y_train, X_test, Y_test, savedir, num_layers=2,
                  num_filters=128, kernel_size=3, pool_size=2, dropout_rate=0.2, learning_rate=0.001,
                  n_epochs=100, batch_size=96, verbose=1):

    model = Sequential()
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _ in range(num_layers):
        model.add(Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    history = model.fit(X_train, Y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        verbose=verbose)
    model.summary()

    scores = model.evaluate(X_test, Y_test)
    print("Loss: %.2f%%" % (scores[0]))
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    model_path = os.path.join(savedir, 'saved_model_' + date_time + '.h5')  # Saving as HDF5 file
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model

# Takes in a list of images and model, calcs the uncertainty and outputs a list of uncertainties
def calculate_uncertainty(model, test_images): # TODO Maybe make it output a tuple with image name and its uncertainty
    uncertainties = []
    for image in test_images:
        prediction = model.predict(image)
        # Calculate uncertainty based on prediction probabilities
        uncertainty = np.mean(np.var(prediction, axis=1))
        uncertainties.append(uncertainty)
    return uncertainties