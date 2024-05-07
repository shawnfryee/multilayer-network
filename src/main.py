import os
import retrain
import train
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

# Define functions for evaluation metrics

def calculate_metrics(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    return conf_matrix, f1, recall, precision, accuracy

def plot_roc_curve(true_labels, predicted_probabilities):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def plot_accuracy_loss(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_log_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

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


def load_data(image_folder, target_size=(224, 224)):
    batch_size = 32


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

    print(train_ds.class_names)

    return train_ds, val_ds


def entropy(predictions):
    return -np.sum(predictions * np.log(predictions + 1e-5), axis=1)  # Adding a small number to avoid log(0)

def uncertain_samples(val_ds, model, percentile=50):
    all_preds = []
    for images, _ in val_ds:
        preds = model.predict(images)
        all_preds.extend(preds)

    uncertainties = entropy(np.array(all_preds))
    threshold = np.percentile(uncertainties, percentile)

    uncertain_images = []
    uncertain_labels = []

    for images, labels in val_ds:
        preds = model.predict(images)
        batch_uncertainties = entropy(preds)
        for img, label, uncertainty in zip(images, labels, batch_uncertainties):
            if uncertainty > threshold:
                uncertain_images.append(img.numpy())
                uncertain_labels.append(label.numpy())

    return np.array(uncertain_images), np.array(uncertain_labels)

def main():
    output_directory = make_output_directory()
    #log_file_path = os.path.join(output_directory, 'output_log.txt')
    #sys.stdout = open(log_file_path, 'w')

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

    num_classes = 4 # stop, continue, right, left

    #setup training stuff
    train_x, test_x = load_data(data_folder)

    model = train.fit_model(train_x, test_x, output_directory)

    # Evaluate model on validation set
    true_labels = []
    predicted_probabilities = []
    for images, labels in test_x:
        true_labels.extend(labels.numpy())
        preds = model.predict(images)
        predicted_probabilities.extend(preds)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    # Calculate metrics
    conf_matrix, f1, recall, precision, accuracy = calculate_metrics(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("F1 Score:", f1)
    print("Recall (R2):", recall)
    print("Precision:", precision)
    print("Accuracy:", accuracy)

    # Plot ROC curve
    plot_roc_curve(true_labels, np.array(predicted_probabilities))

    # Uncertainty estimation and filtering
    uncertain_imgs, uncertain_labels = uncertain_samples(train_x, model, percentile=5) #retrain with top 5% (least certain)
    print("uncertain images: {}", uncertain_imgs)
    uncertain_ds = tf.data.Dataset.from_tensor_slices((uncertain_imgs, uncertain_labels)).batch(32)
    retrain_history = retrain.finetune_model(model, uncertain_ds, test_x, output_directory)

    # Plot accuracy and loss
    plot_accuracy_loss(retrain_history)
    plot_log_loss(retrain_history)


if __name__ == "__main__":
    main()