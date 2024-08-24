# Image Classification with Enhanced ResNet Models Using SE Attention

This project implements a ResNet-based image classification model enhanced with Squeeze-and-Excitation (SE) blocks to improve feature representation. The model is trained and evaluated using a dataset of animal images, leveraging Google Colab for computation and storage.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Prediction](#prediction)
- [Results](#results)
- [License](#license)

## Overview

The project demonstrates the training and evaluation of a ResNet model enhanced with SE Attention. It uses a dataset containing various classes of animal images, which are preprocessed and split into training and validation sets. The model's performance is evaluated based on accuracy and confusion matrix metrics.

## Requirements

To run the project, you'll need the following libraries:

- Python 3.x
- Google Colab (or a similar environment with GPU support)
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

You can install the necessary libraries via pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## Project Structure

The main components of the project are as follows:

- **Data Preparation**: Splits the dataset into training and validation sets.
- **Model Definition**: Defines the ResNet model with SE Attention blocks.
- **Training Script**: Trains the model using the prepared dataset.
- **Evaluation Script**: Evaluates the model on the validation set.
- **Prediction Script**: Performs inference on new images.

## Setup

1. **Mount Google Drive**: The project assumes that you're using Google Colab. Start by mounting your Google Drive to access datasets and save model weights.
   
2. **Prepare Dataset**:
   - Ensure your dataset is in your Google Drive in a directory named `dataset_42028assg2_24708935`.
   - The dataset should be unzipped and contain subdirectories for each class of animals.

3. **Model Files**:
   - Download the pre-trained weights (`resnet34-pre.pth`) and `class_indices.json` from your Google Drive.

## Usage

### Data Preparation

Run the following command to copy and prepare the dataset:

```python
!cp -r /content/gdrive/MyDrive/dataset_42028assg2_24708935/ ./dataset
```

Then, use the script to split the dataset into training and validation sets:

```python
python split_dataset.py
```

### Training the Model

Train the model using the `train.py` script:

```python
python train.py
```

This script:

- Loads and preprocesses the dataset.
- Defines the ResNet model with SE Attention.
- Compiles the model and sets the optimizer and loss function.
- Trains the model over several epochs, logging training loss and validation accuracy.

### Evaluating the Model

Evaluate the trained model using the validation set:

```python
python evaluate.py
```

The evaluation script:

- Loads the trained model weights.
- Computes the accuracy on the validation set.
- Generates a confusion matrix to visualize performance.

### Prediction

To make predictions on new images, use the `predict.py` script:

```python
python predict.py
```

The script:

- Loads the model and weights.
- Preprocesses the input image.
- Outputs the predicted class and probability.

### Example Commands

To train the model:

```bash
python resnet_train.py
```

To evaluate the model:

```bash
python resnet_evaluate.py
```

To perform prediction on a single image:

```bash
python resnet_predict.py
```

## Results

- **Training Loss and Validation Accuracy**: The training script plots the loss curve and accuracy curve over the training epochs.
- **Confusion Matrix**: The evaluation script generates a confusion matrix to visualize model performance across different classes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
