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

    # Customize x-tick labels
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=0, ha='center')

    plt.tight_layout()
    plt.show()
    