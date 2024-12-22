import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_models(pipeline_ORB, pipeline_SIFT, df):
    """
    Evaluate the accuracy of two models (ORB and SIFT) on a test dataset and return a DataFrame of the accuracies.

    Parameters:
    pipeline_ORB (Pipeline): The ORB model pipeline.
    pipeline_SIFT (Pipeline): The SIFT model pipeline.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test labels.

    Returns:
    pd.DataFrame: A DataFrame containing the accuracies of both models.
    """
    # Make predictions with ORB model
    y_test = df['Target']
    y_pred_ORB = pipeline_ORB.predict(df)
    accuracy_ORB = accuracy_score(y_test, y_pred_ORB)

    # Make predictions with SIFT model
    y_pred_SIFT = pipeline_SIFT.predict(df)
    accuracy_SIFT = accuracy_score(y_test, y_pred_SIFT)

    # Create a DataFrame with the accuracies
    accuracy_data = {
        'Model': ['ORB', 'SIFT'],
        'Accuracy': [accuracy_ORB, accuracy_SIFT]
    }
    accuracy_df = pd.DataFrame(accuracy_data)

    return accuracy_df
