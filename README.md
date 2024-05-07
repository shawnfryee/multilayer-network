# Multilayer Networking Graduate Project

## Introduction
This repository contains the code and documentation for the Multilayer Networking Graduate Project conducted by Kyatt Spessert, Danielle Ware, Luke Redwine, Shawn Frye, and Jerrick Dubose for CSE-6693.

## Overview
The project focuses on exploring the applications and implementations of multilayer networking techniques in the context of graduate-level research.

## Contributors
- Kyatt Spessert
- Danielle Ware
- Luke Redwine
- Shawn Frye
- Jerrick Dubose

## Contents
- `src/`: Contains the implementation code for the project.
- `docs/`: Contains project documentation, including research papers, presentations, and technical reports.

## Configuration Parameters

### Epochs
- **Parameter name:** `epochs`
- **Description:** The number of epochs to train the model for.
- **Type:** Integer


### Learning Rate
- **Parameter name:** `learning_rate`
- **Description:** The learning rate used by the optimizer during training.
- **Type:** Float


### Batch Size
- **Parameter name:** `batch_size`
- **Description:** The number of samples per batch used for training.
- **Type:** Integer


## Example Configuration File

```ini
# Example config.ini

[Training]
epochs = 10
learning_rate = 0.001
batch_size = 32

```

## Running the Code

Follow these steps to run the application:

1. **Install Dependencies**:
   - Open your terminal or command prompt.
   - Navigate to the main project directory.
   - Run the command `pip install -r requirements.txt` to install all required packages.

2. **Prepare the Dataset**:
   - Ensure that you have the zipped dataset files downloaded.
   - Decompress all zipped files into one folder.
   - Place the decompressed files into a directory named `dataset/` located in the main project directory.

3. **Run the Application**:
   - Continue in the terminal or command prompt.
   - Run the command `python main.py` to start the application.

Make sure you have Python installed on your machine. If not, you can download it from [python.org](https://www.python.org/downloads/).
