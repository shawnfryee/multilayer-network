import os
import configparser

from tensorflow.python.keras.utils import np_utils

import train
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import random_rotation, random_brightness


def make_output_directory():
    current_project_directory = os.getcwd()
    output_directory = os.path.join(os.path.dirname(current_project_directory), "output")
    # Check if the directory exists
    if not os.path.exists(output_directory):
        # Create the directory if it doesn't exist
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created successfully.")
    else:
        print(f"Directory '{output_directory}' already exists.")
    return output_directory


def load_data(image_folder, label_folder, target_size=(224, 224)):
    batch_size = 32
    image_height = target_size[1]
    image_width = target_size[0]

    train_ds = image_dataset_from_directory(
        image_folder,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )

    val_ds = image_dataset_from_directory(
        image_folder,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )

    # new_train = random_brightness(train_ds, (0, 30))
    # new_val = random_brightness(val_ds, (0, 30))
    #
    # new_train = random_rotation(new_train, 36)
    # new_val = random_rotation(new_val, 36)

    print(train_ds.class_names)

    return train_ds, val_ds


def load_data_sean(image_folder, label_folder, target_size=(224, 224)):
    # Load and sort dataset
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)
    image_files.sort()
    label_files.sort()

    images = []
    labels = []

    # Load images and labels
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)

        label_file = img_file.replace(".jpg", ".txt").replace(".jpeg", ".txt")
        label_path = os.path.join(label_folder, label_file)

        img = Image.open(img_path).resize(target_size)
        img_array = np.array(img) / 255.0
        images.append(img_array)

        with open(label_path, 'r') as f:
            label_str = f.readline()#.strip()
            label_dict = {'stop': 0, 'continue': 1, 'left': 2, 'right': 3}
            label = label_dict.get(label_str, -1)
            if label == -1:
                print(label)
                raise ValueError("Unknown label: {}".format(label_str))

        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)


    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, Y_train, X_test, Y_test


def main():
    output_directory = make_output_directory()
# TODO make a config file that is read to change number of epochs and different parameters for optimizing ther model
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if tf.config.experimental.list_physical_devices('GPU'):
        print("TensorFlow will be able to use the GPU!")
    else:
        print(
            "Make sure your system meets all requirements and the TensorFlow version is compatible with the installed CUDA and cuDNN.")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    #get dataset
    data_folder = os.path.join(os.path.dirname(os.getcwd()), "dataset/")
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")
    num_classes = 4 # stop, continue, right, left

    #setup training stuff
    train_x, test_x = load_data(data_folder, labels_folder)
    # Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=num_classes)
    # Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=num_classes)
    # X_train = X_train.astype(np.float32)
    # X_test = X_test.astype(np.float32)
    # model = train.fit_cnn_model(X_train, Y_train, X_test, Y_test, output_directory)
    train.fit_mode(train_x, test_x, output_directory)

    # for batch in train_x.take(4):
    #     print(batch[1])


if __name__ == "__main__":
    main()