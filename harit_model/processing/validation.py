import os
from PIL import Image

def validate_and_resize_dataset(dataset_path, expected_shape=(224, 224, 3)):
    """
    Validate and resize images in the dataset directory to the expected shape.
    Args:
        dataset_path (str): Path to the dataset directory.
        expected_shape (tuple): Expected image shape (height, width, channels).
    Returns:
        None
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' is not a directory.")
        return

    total_files = 0
    invalid_files = 0
    incorrect_shape_files = 0
    resized_files = 0
    classes = os.listdir(dataset_path)

    print(f"Validating and resizing dataset at: {dataset_path}")
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Skipping non-directory item: {class_name}")
            continue

        print(f"Validating class: {class_name}")
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            total_files += 1

            # Check if the file is an image and its shape
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it's an image

                    # Reopen to check the image's actual shape
                    img = Image.open(file_path)
                    if img.size != expected_shape[:2]:
                        print(f"Incorrect shape: {file_path}, Found: {img.size}, Expected: {expected_shape[:2]}")
                        incorrect_shape_files += 1
                        # Resize the image to the expected shape
                        img = img.resize(expected_shape[:2], Image.ANTIALIAS)
                        img.save(file_path)  # Save the resized image
                        resized_files += 1

            except Exception as e:
                print(f"Invalid image: {file_path}, Error: {e}")
                invalid_files += 1

    print("\nValidation and Resizing Summary:")
    print(f"Total files checked: {total_files}")
    print(f"Invalid files: {invalid_files}")
    print(f"Files with incorrect shape: {incorrect_shape_files}")
    print(f"Files resized: {resized_files}")
    if invalid_files == 0 and incorrect_shape_files == 0:
        print("Dataset is valid and resized!")
    else:
        print("Some issues were found. Please review the errors above.")

# Replace this with your dataset directory path
dataset_directory = r"C:\Akshay\AIMLOps24\Capstone Project\CapstoneProject-Group3\harit_model\datasets"
validate_and_resize_dataset(dataset_directory)
