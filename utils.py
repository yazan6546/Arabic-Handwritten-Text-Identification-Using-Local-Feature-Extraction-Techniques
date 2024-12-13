import os  # OS module for file operations
import shutil  # Shutil for file operations

def extract_words(directory):
    words = set()
    for root, _, files in os.walk(directory):
        for file in files:
            parts = file.split('_')
            if len(parts) > 2:
                words.add(parts[1])
    return words

def convert_to_arabic(word_set):
    list_words = ['غزال', 'شطيرة', 'فسيكفيكهم', 'قشطة', 'صخر', 'اذن', 'مستهدفين', 'محراس', 'غليظ', 'ابجدية']
    dict_words = {key:word for key, word in zip(word_set, list_words)}

    return dict_words



def collect_images_and_copy_to_original(source_directory, target_directory):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

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


