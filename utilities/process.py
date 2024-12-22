import os
import pandas as pd
from sklearn.metrics import accuracy_score
import utils


def process_modify_directory(directory, pipeline_sift, pipeline_orb):

    results = []

    type = os.path.basename(directory)

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            # Load the dataframe from the subdirectory
            df = utils.load_images_to_dataframe(subdir_path)

            # Predict and calculate accuracy for SIFT
            y_pred_sift = pipeline_sift.predict(df)
            accuracy_sift = accuracy_score(df["Target"], y_pred_sift)

            # Predict and calculate accuracy for ORB
            y_pred_orb = pipeline_orb.predict(df)
            accuracy_orb = accuracy_score(df['Target'], y_pred_orb)

            # Append results to the list
            results.append(
                {
                    "transformation": f"{type}_{subdir}",
                    "accuracy_sift": accuracy_sift,
                    "accuracy_orb": accuracy_orb,
                }
            )
        return pd.DataFrame(results)


def process_output_directory(directory, pipeline_sift, pipeline_orb):

    dictionary = {}
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            dictionary[subdir] = process_modify_directory(subdir_path, pipeline_sift, pipeline_orb)

    return dictionary