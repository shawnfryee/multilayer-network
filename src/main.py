import os
import configparser
import train
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


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


def load_data(image_folder, label_folder):
    # load n sort dataset TODO needs to be expanded for subdirs for stop, go, left, right. This is just very basic
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)
    image_files.sort()
    label_files.sort()

    images = []
    labels = []

    # Load images and labels
    for img_file, label_file in zip(image_files, label_files):
        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, label_file)

        img = Image.open(img_path)
        img_array = np.array(img)
        images.append(img_array)

        # TODO needs to be expanded for stop, go, left, right labels. This is just very basic
        with open(label_path, 'r') as f:
            label = int(f.readline().strip())  # Assuming label is a single digit
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, Y_train, X_test, Y_test
def main():
    output_directory = make_output_directory()
# TODO make a config file that is read to change number of epochs and different parameters for optimizing ther model
    #get dataset
    data_folder = os.path.join(os.path.dirname(os.getcwd()), "dataset/")
    images_folder = os.path.join(data_folder, "images")
    labels_folder = os.path.join(data_folder, "labels")

    #setup training stuff
    X_train, Y_train, X_test, Y_test = load_data(images_folder, labels_folder)
    model = train.fit_cnn_model(X_train, Y_train, X_test, Y_test, output_directory)


if __name__ == "__main__":
    main()