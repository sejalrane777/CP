import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Global variables to store datasets
train_data = None
valid_data = None
test_data = None

# Specify the output path for the dataset
try:
    # Attempt to download the dataset
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    print("Dataset downloaded successfully to:", path)
except Exception as e:
    print("Error downloading dataset:", str(e))

import random
import os
import warnings
warnings.filterwarnings('ignore')

def dataset():
    """
    Loads datasets only if they haven't been loaded already.
    Returns train_data, valid_data, and test_data.
    """
    global train_data, valid_data, test_data

    # if train_data is None or valid_data is None or test_data is None:
    print("Loading datasets...")
    image_shape = (224, 224)
    batch_size = 30

    train_dir = path + "/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    valid_dir = path + "/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

    # Apply scaling only because data is already augmented
    train_datagen = ImageDataGenerator(rescale=1 / 255., validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    # Load training data
    print("Training Images:")
    sample_fraction = 0.1
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training')
    indices = random.sample(range(len(train_data.filenames)), int(len(train_data.filenames) * sample_fraction))
    train_data = [train_data.filenames[i] for i in indices]

    # Load validation data (20% of training data)
    print("Validating Images:")
    valid_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation')

    # Load test data (consider validation data as test data)
    print("Test Images:")
    test_data = test_datagen.flow_from_directory(
        valid_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
# else:
#     print("Datasets already loaded.")

    return train_data, valid_data, test_data
#if condtion for it will not load dataset again hile predicting
def class_names():
    """
    Retrieves class names from the test dataset.
    """
    global test_data
    if test_data is None:
        _, _, test_data = dataset()  # Ensure test_data is loaded
    return list(test_data.class_indices.keys())
