import os  # OS module for file operations
import shutil  # Shutil for file operations
import cv2


# Load Images from File
def load_images_from_directory(directory, target_size=(256, 128)):
    data = []
    for root, _, files in os.walk(directory):
            
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = cv2.imread(os.path.join(root, filename), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                            
                    data.append({'filename': filename, 'image': img})
    return data


def collect_images_and_copy_to_original(source_directory, target_directory):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    if not os.path.exists(source_directory):
        raise FileNotFoundError(f"Source directory {source_directory}")

    # Traverse all directories in the source directory
    for root, dirs, files in os.walk(source_directory, topdown=False):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png"):
                # Construct the full file path
                file_path = os.path.join(root, name)
                
                # Copy the file to the target directory
                shutil.move(file_path, target_directory)
                print(f"Copied {file_path} to {target_directory}")

    # Delete the entire source directory
    shutil.rmtree(source_directory)



def preprocess_images(source_directory, target_directory, threshold_value=200, target_size=(256, 128)):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Traverse all directories in the source directory
    for root, dirs, files in os.walk(source_directory, topdown=False):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png"):
                # Construct the full file path
                file_path = os.path.join(root, name)
                
                # Load the image in grayscale
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize the image
                resized_image = cv2.resize(image, target_size)
                
                # Apply global thresholding
                _, thresholded_image = cv2.threshold(resized_image, threshold_value, 255, cv2.THRESH_BINARY)
                
                # Construct the target file path
                target_file_path = os.path.join(target_directory, name)
                
                # Save the thresholded image to the target directory
                cv2.imwrite(target_file_path, thresholded_image)
                
                # print(f"Processed and saved {file_path} to {target_file_path}")
