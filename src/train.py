import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import utils
from datetime import datetime

def fit_cnn_model(X_train, Y_train, X_test, Y_test, savedir, num_layers=2,
                  num_filters=64, kernel_size=3, pool_size=2, dropout_rate=0.2, learning_rate=0.001,
                  n_epochs=50, batch_size=32, verbose=1):

    model = Sequential()
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _ in range(num_layers):
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(Y_train.shape[0], activation='softmax'))

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        verbose=verbose)

    model.summary()

    scores = model.evaluate(X_test, Y_test)
    print("Loss: %.2f%%" % (scores[0]))
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    utils.save_model(model, os.path.join(savedir, 'savd_model_' + date_time))

    return model
