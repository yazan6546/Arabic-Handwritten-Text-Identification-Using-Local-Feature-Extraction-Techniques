import os
import cv2
import numpy as np

def apply_rotations_and_save(input_dir=os.path.join('data', 'preprocessed'), output_dir=os.path.join('data', 'rotate'), rotation_angles=[45, 90, 135]):
    # Traverse each image in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Apply rotations and save
                for angle in rotation_angles:
                    # Get the image center and dimensions
                    h, w = img.shape
                    center = (w // 2, h // 2)

                    # Calculate the rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                    # Compute the new bounding box dimensions to prevent cropping
                    abs_cos = abs(rotation_matrix[0, 0])
                    abs_sin = abs(rotation_matrix[0, 1])
                    new_w = int(2 * h * abs_sin + w * abs_cos)
                    new_h = int(2 * h * abs_cos + w * abs_sin)

                    # Adjust the rotation matrix to account for the translation
                    rotation_matrix[0, 2] += (new_w / 2) - center[0]
                    rotation_matrix[1, 2] += (new_h / 2) - center[1]

                    # Perform the rotation with the new bounding box size
                    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

                    # Resize the rotated image back to the original dimensions
                    resized_img = cv2.resize(rotated_img, (w, h))

                    # Save the image in an angle-based folder
                    angle_dir = os.path.join(output_dir, f"{angle}")
                    os.makedirs(angle_dir, exist_ok=True)

                    output_path = os.path.join(angle_dir, filename)
                    cv2.imwrite(output_path, resized_img)

    print(f"Images saved to {output_dir}")



def apply_noise_and_save(input_dir=os.path.join('data', 'preprocessed'), output_dir=os.path.join('data', 'noise'), noise_levels=[10, 50]):
    # Traverse each image in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Apply noise and save
                for noise_level in noise_levels:
                    # Generate Gaussian noise
                    noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)

                    # Add the noise to the image
                    noisy_img = cv2.add(img, noise)

                    # Save the noisy image in a noise-level-based folder
                    noise_dir = os.path.join(output_dir, f"noise_{noise_level}")
                    os.makedirs(noise_dir, exist_ok=True)

                    output_path = os.path.join(noise_dir, filename)
                    cv2.imwrite(output_path, noisy_img)

    print(f"Noisy images saved to {output_dir}")





def scale_image(image, scale):
    # Get the image dimensions
    h, w = image.shape[:2]

    # Calculate the new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return scaled_img




def apply_scaling_and_save(input_dir=os.path.join('data', 'preprocessed'), output_dir=os.path.join('data', 'scaling'), scaling_factors=[0.5, 0.75, 1.25, 1.5]):
    # Traverse each image in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Apply scaling and save
                for scale in scaling_factors:
                    # Apply scaling
                    scaled_img = scale_image(img, scale)

                    # Save the scaled image in a scale-based folder
                    scale_dir = os.path.join(output_dir, f"scale_{str(scale).replace('.', '_')}")
                    os.makedirs(scale_dir, exist_ok=True)

                    output_path = os.path.join(scale_dir, filename)
                    cv2.imwrite(output_path, scaled_img)

    print(f"Scaled images saved to {output_dir}")


def modify_images(input_dir=os.path.join('data', 'preprocessed'), output_dir=os.path.join('data', 'output'), rotation_angles=[45, 90, 135], noise_levels=[10, 20, 30], scaling_factors=[0.5, 0.75, 1.25, 1.5]):
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    rotate_dir = os.path.join(output_dir, "rotate")
    noise_dir = os.path.join(output_dir, "noise")
    scaling_dir = os.path.join(output_dir, "scaling")

    # Apply rotations and save
    apply_rotations_and_save(input_dir=input_dir, output_dir=rotate_dir, rotation_angles=rotation_angles)

    # Apply noise and save
    apply_noise_and_save(input_dir=input_dir, output_dir=noise_dir, noise_levels=noise_levels)

    # Apply scaling and save
    apply_scaling_and_save(input_dir=input_dir, output_dir=scaling_dir, scaling_factors=scaling_factors)

    print(f"Preprocessing completed. Images saved to {output_dir}")

if __name__ == "__main__":
    modify_images()