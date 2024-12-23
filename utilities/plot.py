import os
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd 


def plot_random_images(original_dir, preprocessed_dir, num_images=2, seed=42):
    # Get list of filenames in both directories
    random.seed(seed)

    
    original_filenames = set(os.listdir(original_dir))
    preprocessed_filenames = set(os.listdir(preprocessed_dir))

    # Find common filenames
    common_filenames = list(original_filenames.intersection(preprocessed_filenames))

    # Select random filenames
    selected_filenames = random.sample(common_filenames, num_images)



    # Plot the images
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    # Add large subtitles
    fig.suptitle('Comparison of Original and Preprocessed Images', fontsize=16)
    fig.text(0.25, 0.92, 'Original', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.92, 'Preprocessed', ha='center', fontsize=14, fontweight='bold')
    for i, filename in enumerate(selected_filenames):
        # Load original image
        original_image_path = os.path.join(original_dir, filename)
        original_image = cv2.imread(original_image_path)

        # Load preprocessed image
        preprocessed_image_path = os.path.join(preprocessed_dir, filename)
        preprocessed_image = cv2.imread(preprocessed_image_path)

        # Plot original image
        axes[i, 0].imshow(original_image, cmap='gray')

        # Plot preprocessed image
        axes[i, 1].imshow(preprocessed_image, cmap='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0, h_pad=1.0, w_pad=1.0)
    plt.show()


def plot_grouped_barcharts(df):
    # Plot grouped bar chart for accuracy
    accuracy_df = df[['accuracy_orb', 'accuracy_sift']]
    ax = accuracy_df.plot(kind='bar', figsize=(10, 6), title='Accuracy for ORB and SIFT Pipelines')
    plt.xlabel('Transformation Type')
    plt.ylabel('Accuracy')

    # Adjust legend position
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    plt.show()
    

# Example usage
# Assuming results_dict is a dictionary of DataFrames with the required columns
results_dict = {
    'noise_10': pd.DataFrame({
        'accuracy_orb': [0.85],
        'accuracy_sift': [0.88]
    }),
    'noise_20': pd.DataFrame({
        'accuracy_orb': [0.86],
        'accuracy_sift': [0.89]
    }),
    'scale_1.5': pd.DataFrame({
        'accuracy_orb': [0.80],
        'accuracy_sift': [0.83]
    }),
    'scale_2.0': pd.DataFrame({
        'accuracy_orb': [0.81],
        'accuracy_sift': [0.84]
    }),
    'rotation_30': pd.DataFrame({
        'accuracy_orb': [0.78],
        'accuracy_sift': [0.81]
    }),
    'rotation_60': pd.DataFrame({
        'accuracy_orb': [0.79],
        'accuracy_sift': [0.82]
    })
}

plot_grouped_barcharts(results_dict)