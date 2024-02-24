import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import utils

def fit_cnn_model(X_train, Y_train, X_test, Y_test, savedir, combination, num_layers=2,
                  num_filters=64, kernel_size=3, pool_size=2, dropout_rate=0.2, learning_rate=0.001,
                  n_epochs=50, batch_size=32, verbose=1):

    model = Sequential()

    for _ in range(num_layers):
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(Y_train.shape[1], activation='softmax'))

    opt = Adam(lr=learning_rate)
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

    plt.figure(figsize=(10, 10))
    plt.plot(history.history['accurac'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join(savedir, 'model_accuracy_' + combination + '.png'))

    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Los')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(os.path.join(savedir, 'model_loss_' + combination + '.png'))

    utils.save_model(model, os.path.join(savedir, 'savd_model_' + combination))

    return model
