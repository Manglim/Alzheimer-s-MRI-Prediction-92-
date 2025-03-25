# Alzheimer’s Disease MRI Classification

## Overview
This project is a deep learning-based application designed to classify MRI scans for Alzheimer’s disease impairment levels. It leverages a pre-trained ResNet-50 convolutional neural network (CNN) to process and classify MRI images into four categories: "Mild Impairment," "Moderate Impairment," "No Impairment," and "Very Mild Impairment." The project includes data preprocessing, model training, evaluation, and a prediction function for new images. It is developed in a Kaggle environment, utilizing a specific dataset for Alzheimer’s MRI scans.

## Functionality
1. **Data Preprocessing and Loading**:
   - Loads a dataset from `/kaggle/input/best-alzheimer-mri-dataset-99-accuracy/Combined Dataset` with `train` and `test` subdirectories.
   - Applies image transformations (resize, augmentation, normalization) to prepare data for the model.
   - Uses PyTorch’s `ImageFolder` and `DataLoader` to organize and batch the data.

2. **Model Setup**:
   - Utilizes a pre-trained ResNet-50 model from `torchvision.models`.
   - Modifies the final fully connected layer to match the number of classes (4).
   - Supports GPU acceleration if available (`cuda`), otherwise runs on CPU.

3. **Training**:
   - Trains the model for 20 epochs using the Adam optimizer and CrossEntropyLoss.
   - Implements a learning rate scheduler (`ReduceLROnPlateau`) to adjust the learning rate based on validation loss.
   - Tracks training and validation loss, as well as precision, saving the model with the highest precision.

4. **Evaluation**:
   - Evaluates the model on the test dataset after each epoch, calculating precision using `sklearn.metrics.precision_score`.
   - Saves the best-performing model to `best_alzheimer_model.pth`.

5. **Prediction**:
   - Provides a function (`predict_impairment`) to classify new MRI images using the trained model.
   - Outputs the predicted class and confidence score for a given image.

## Frameworks and Libraries
- **Python**: Core programming language (Python 3).
- **PyTorch**: Primary deep learning framework (`torch`, `torchvision`, `torchaudio`).
  - `torch`: Core tensor operations and neural network functionality.
  - `torchvision`: Pre-trained models (ResNet-50), datasets (`ImageFolder`), and transformations (`transforms`).
  - `torch.utils.data.DataLoader`: Handles batching and shuffling.
- **NumPy**: Numerical operations (`np`).
- **Pandas**: Included but minimally used (`pd`).
- **OS**: File system operations (e.g., `os.walk`).
- **PIL (Pillow)**: Image loading and manipulation (`Image`).
- **Scikit-learn**: Precision metric calculation (`precision_score`).
- **Kaggle Environment**: Runs in a Kaggle Docker environment with pre-installed libraries.

## Dataset
- Uses the ["Best Alzheimer MRI Dataset 99% Accuracy"](https://www.kaggle.com/datasets/best-alzheimer-mri-dataset-99-accuracy) from Kaggle.
- Organized into `train` and `test` folders, with subfolders for each class (e.g., "No Impairment," "Mild Impairment").
- Assumes images are in a format compatible with `ImageFolder` (e.g., `.jpg`).

## Key Features
- **Data Augmentation**: Applies `RandomHorizontalFlip` and `RandomRotation` to training images.
- **Normalization**: Uses ImageNet mean and standard deviation for consistency with ResNet-50.
- **Model Checkpointing**: Saves the model with the highest precision.
- **Performance Metrics**: Tracks training loss, validation loss, and weighted precision.
- **Hardware Optimization**: Automatically utilizes GPU if available.

## Execution Flow
1. **Setup**: Imports libraries, defines paths, and sets up data transformations and loaders.
2. **Model Configuration**: Loads ResNet-50, adjusts the final layer, and moves it to the appropriate device.
3. **Training Loop**: Iterates over 20 epochs, training and evaluating the model.
4. **Prediction**: Loads the best model and provides a function to classify new images.

## Sample Output
- Training progresses over 20 epochs, with precision improving from 0.5728 to 0.9280.
- Example prediction: "Mild Impairment" with a confidence of 0.9939 for a sample image.

## Installation
To run this project, ensure the following dependencies are installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn pillow
