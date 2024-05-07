from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import datetime


def finetune_model(pretrained_model, train, test, savedir,
                   learning_rate=0.0001, n_epochs=5, batch_size=96, verbose=1):

    pretrained_model.pop()  # Remove the last softmax layer
    pretrained_model.pop()  # Remove the second last dense layer
    for layer in pretrained_model.layers:
        layer.trainable = False  # Freeze the weights of the original layers

    # Adding new trainable layers
    pretrained_model.add(Dense(64, activation='relu'))
    pretrained_model.add(Dense(4, activation='softmax'))

    # Compile model
    opt = Adam(learning_rate=learning_rate)
    pretrained_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    callback = EarlyStopping(monitor='loss', patience=3)

    # Training the model
    history = pretrained_model.fit(train, validation_data=test,
                                   epochs=n_epochs,
                                   batch_size=batch_size,
                                   verbose=verbose,
                                   callbacks=[callback])

    pretrained_model.summary()

    print(history.history)

    scores = pretrained_model.evaluate(test)
    print("Loss: %.2f%%" % (scores[0]))
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # Save the fine-tuned model
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_path = os.path.join(savedir, 'finetuned_model_' + date_time + '.h5')  # Saving as HDF5 file
    pretrained_model.save(model_path)
    print(f"Model saved to {model_path}")

    return pretrained_model
