# Image Classification with Enhanced ResNet Models Using SE Attention

This project features a ResNet-based image classification model enhanced with Squeeze-and-Excitation (SE) blocks to improve feature representation. The model is developed and trained using a dataset of animal images, with the computation and storage facilitated by Google Colab.

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

This project demonstrates how to train and evaluate a ResNet model enhanced with SE Attention mechanisms. The dataset consists of various animal classes, which are preprocessed and divided into training and validation sets. The model's performance is assessed using accuracy metrics and confusion matrices.

## Requirements

To run this project, ensure you have the following libraries installed:

- Python 3.x
- Google Colab (or a similar environment with GPU support)
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

Install these dependencies using pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## Project Structure

The project is organized into several key components:

- **Data Preparation**: Handles the splitting of the dataset into training and validation sets.
- **Model Definition**: Defines the ResNet model architecture, incorporating SE Attention blocks.
- **Training Script**: Trains the model using the prepared dataset.
- **Evaluation Script**: Evaluates the model's performance on the validation set.
- **Prediction Script**: Provides inference capabilities for new images.

## Setup

1. **Mount Google Drive**: The project assumes you're using Google Colab. Start by mounting your Google Drive to access the datasets and save model weights.
   
2. **Prepare Dataset**:
   - Place your dataset in your Google Drive.
   - Ensure the dataset is unzipped and organized into subdirectories representing each animal class.

3. **Model Files**:
   - Download the pre-trained weights like (`resnet34-pre.pth`) and `class_indices.json` from Google Drive to use them in training and evaluation. (just a example)

## Usage

### Data Preparation

First, copy the dataset from Google Drive to your Colab environment:

```python
!cp -r /content/gdrive/MyDrive/dataset_42028assg2_24708935/ ./dataset
```

Then, execute the script to split the dataset into training and validation sets:

```python
python split_dataset.py
```

### Training the Model

Train the model by running the `train.py` script:

```python
python train.py
```

This script will:

- Load and preprocess the dataset.
- Define the ResNet model with SE Attention.
- Set up the optimizer and loss function.
- Train the model over several epochs, logging training loss and validation accuracy.

### Evaluating the Model

To evaluate the model's performance, run the evaluation script:

```python
python evaluate.py
```

This script:

- Loads the trained model weights.
- Computes the accuracy on the validation set.
- Generates a confusion matrix to visualize class-specific performance.

### Prediction

For making predictions on new images, use the `predict.py` script:

```python
python predict.py
```

This script will:

- Load the model and trained weights.
- Preprocess the input image.
- Output the predicted class and the associated probability.

### Example Commands

Train the model:

```bash
python resnet_train.py
```

Evaluate the model:

```bash
python resnet_evaluate.py
```

Make a prediction on a single image:

```bash
python resnet_predict.py
```

## Results

- **Training Loss and Validation Accuracy**: The training script provides plots for loss and accuracy across training epochs.
- **Confusion Matrix**: The evaluation script generates a confusion matrix, offering a visual insight into the model's performance across different animal classes.

