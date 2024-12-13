import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import warnings
warnings.filterwarnings('ignore')

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

# Function to reduce dataset size by sampling
def create_subset_data(data_gen, directory, sample_fraction=0.1):
    """
    Creates a subset of data from an existing DirectoryIterator.
    Args:
        data_gen: Data generator instance.
        directory: Directory of the dataset.
        sample_fraction: Fraction of data to keep.
    Returns:
        New DirectoryIterator with sampled data.
    """
    # Get all filenames from the generator
    data_gen_original = data_gen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )
    filenames = data_gen_original.filenames
    total_samples = len(filenames)

    # Calculate the number of samples to keep
    num_samples_to_keep = int(total_samples * sample_fraction)

    # Randomly sample filenames
    sampled_filenames = random.sample(filenames, num_samples_to_keep)

    # Create a temporary folder for reduced dataset
    reduced_dir = "reduced_dataset"
    if not os.path.exists(reduced_dir):
        os.makedirs(reduced_dir)

    # Copy sampled files to the reduced dataset folder
    for file_path in sampled_filenames:
        class_name, file_name = file_path.split("/")
        class_dir = os.path.join(reduced_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        src = os.path.join(directory, file_path)
        dst = os.path.join(class_dir, file_name)
        if not os.path.exists(dst):
            os.link(src, dst)  # Use hard links to save space

    # Create a new data generator for the reduced dataset
    reduced_data_gen = data_gen.flow_from_directory(
        reduced_dir,
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical',
        shuffle=True
    )
    return reduced_data_gen

def dataset():
    """
    Loads and processes datasets, including dataset reduction.
    Returns train_data, valid_data, and test_data.
    """
    global train_data, valid_data, test_data

    print("Loading datasets...")
    image_shape = (224, 224)
    batch_size = 64

    train_dir = path + "/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
    valid_dir = path + "/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

    # Apply scaling only because data is already augmented
    train_datagen = ImageDataGenerator(rescale=1 / 255., validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1 / 255.)

    # Create reduced training data
    print("Reducing dataset size...")
    train_data = create_subset_data(train_datagen, train_dir, sample_fraction=0.1)

    # Load validation data
    print("Validating Images:")
    valid_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    # Load test data
    print("Test Images:")
    test_data = test_datagen.flow_from_directory(
        valid_dir,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, valid_data, test_data

def class_names():
    """
    Retrieves class names from the test dataset.
    """
    global test_data
    if test_data is None:
        _, _, test_data = dataset()  # Ensure test_data is loaded
    return list(test_data.class_indices.keys())

# Load datasets
train_data, valid_data, test_data = dataset()

# Check the number of samples in the reduced dataset
print("Number of samples in reduced train dataset:", train_data.samples)
print("Number of classes:", train_data.num_classes)
