import os
from PIL import Image, ImageStat

def is_poor_quality(image_path, threshold=5.0):
    """
    Check if an image is of poor quality based on entropy.
    Args:
        image_path (str): Path to the image.
        threshold (float): Entropy threshold below which an image is considered poor quality.
    Returns:
        bool: True if the image is considered of poor quality, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Ensure the image is in RGB mode
            entropy = img.entropy()  # Calculate entropy of the image
            print(f"Entropy of {image_path}: {entropy:.2f}")

            # If the entropy is below the threshold, consider it poor quality
            if entropy < threshold:
                return True
            return False
    except Exception as e:
        print(f"Error checking image quality for {image_path}: {e}")
        return False

def check_image_quality(dataset_path):
    """
    Check the quality of images in the dataset and print paths of poor-quality images.
    Args:
        dataset_path (str): Path to the dataset directory.
    Returns:
        None
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' is not a directory.")
        return

    poor_quality_images = []
    classes = os.listdir(dataset_path)

    print(f"Checking image quality in dataset: {dataset_path}")
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Skipping non-directory item: {class_name}")
            continue

        print(f"Checking class: {class_name}")
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)

            if is_poor_quality(file_path):
                poor_quality_images.append(file_path)
                print(f"Poor quality image found: {file_path}")

    # Print summary
    print("\nSummary:")
    if poor_quality_images:
        print(f"Total number of poor-quality images found: {len(poor_quality_images)}")
        print("List of poor-quality images:")
        for image in poor_quality_images:
            print(image)
    else:
        print("No poor-quality images found.")

# Replace this with your dataset directory path
dataset_directory = r"C:\Akshay\AIMLOps24\Capstone Project\CapstoneProject-Group3\harit_model\datasets"
check_image_quality(dataset_directory)
