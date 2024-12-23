import os  # OS module for file operations
import shutil  # Shutil for file operations
import cv2
import numpy as np
import pandas as pd


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


def load_images_to_dataframe(directory):
    """
    Load images from a directory and return a DataFrame containing the images and their filenames.

    Parameters:
    directory (str): The directory containing the images.

    Returns:
    pd.DataFrame: A DataFrame with columns 'filename' and 'image'.
    """
    
    data = load_images_from_directory(directory)
    df = pd.DataFrame(data)
    df.set_index('filename', inplace=True)
    df['Target'] = df.index.map(lambda x: x.split('_')[0])

    return df


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


def extract_images(df):
    if 'image' not in df.columns:
        raise KeyError("The DataFrame does not contain an 'image' column.")
    return df['image']


def calculate_average_keypoints(df, image_column):
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()

    orb_keypoints = []
    sift_keypoints = []

    for image in df[image_column]:
        if image is None:
            continue

        # Detect keypoints and descriptors using ORB
        kp_orb, des_orb = orb.detectAndCompute(image, None)
        orb_keypoints.append(len(kp_orb))

        # Detect keypoints and descriptors using SIFT
        kp_sift, des_sift = sift.detectAndCompute(image, None)
        sift_keypoints.append(len(kp_sift))

    # Convert lists to NumPy arrays
    orb_keypoints = np.array(orb_keypoints)
    sift_keypoints = np.array(sift_keypoints)

    sum_orb = np.sum(orb_keypoints) if orb_keypoints.size > 0 else 0
    sum_sift = np.sum(sift_keypoints) if sift_keypoints.size > 0 else 0
    
    image_count = df['image'].shape[0]

    avg_orb_keypoints = sum_orb / image_count
    avg_sift_keypoints = sum_sift / image_count

    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'ORB': {
            'Average_Keypoints': avg_orb_keypoints,
            'Sum_Keypoints': sum_orb
        },
        'SIFT': {
            'Average_Keypoints': avg_sift_keypoints,
            'Sum_Keypoints': sum_sift
        }
    })
    
    return result_df