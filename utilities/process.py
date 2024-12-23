import os
import pandas as pd
from sklearn.metrics import accuracy_score
from utilities import utils
import evaluate

def process_modify_directory(directory, pipeline_sift, pipeline_orb, encoder):
    results = []

    type = os.path.basename(directory)

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir_path}")
            
            # Load the dataframe from the subdirectory
            df = utils.load_images_to_dataframe(subdir_path)
            df['Target'] = encoder.transform(df['Target'])
            print(f"Loaded DataFrame for {subdir}:")
            print(df.head())

            # Ensure the DataFrame has the 'Target' column
            if 'Target' not in df.columns:
                print(f"Error: 'Target' column not found in DataFrame for {subdir}")
                continue

            # Check and convert data types if necessary
            if df['Target'].dtype != 'int64':
                df['Target'] = df['Target'].astype('int64')
                print(f"Converted 'Target' column to int64 for {subdir}")

            # Predict and calculate accuracy for SIFT
            y_pred_sift = pipeline_sift.predict(df)
            accuracy_sift = accuracy_score(df["Target"], y_pred_sift)
            print(f"SIFT Predictions for {subdir}: {y_pred_sift}")
            print(f"SIFT Accuracy for {subdir}: {accuracy_sift}")

            # Predict and calculate accuracy for ORB
            y_pred_orb = pipeline_orb.predict(df)
            accuracy_orb = accuracy_score(df["Target"], y_pred_orb)
            print(f"ORB Predictions for {subdir}: {y_pred_orb}")
            print(f"ORB Accuracy for {subdir}: {accuracy_orb}")

            # Append results to the list
            results.append({
                "transformation": f"{type}_{subdir}",
                "accuracy_sift": accuracy_sift,
                "accuracy_orb": accuracy_orb,
            })

    return pd.DataFrame(results)


def process_output_directory(directory, pipeline_sift, pipeline_orb, encoder):

    dictionary = {}
    for subdir in os.listdir(directory):


        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            dictionary[subdir_path] = process_modify_directory(subdir_path, pipeline_sift, pipeline_orb, encoder)

    return dictionary