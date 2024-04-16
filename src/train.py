import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,  Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

from datetime import datetime



def fit_model(train, test, savedir, num_layers=2,
              num_filters=128, kernel_size=3, pool_size=2, dropout_rate=0.2, learning_rate=0.001,
              n_epochs=1, batch_size=96, verbose=1):

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
