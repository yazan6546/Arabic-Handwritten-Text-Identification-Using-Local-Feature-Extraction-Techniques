import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import FunctionTransformer, Pipeline

from classes.feature_extractor import FeatureExtractor

class Clusterer(BaseEstimator, TransformerMixin):
    def __init__(self, num_clusters=100):
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    def fit(self, X, y=None):
        all_descriptors = np.vstack(X.to_numpy())
        self.kmeans.fit(all_descriptors)
        return self

    def transform(self, X):
        def create_histogram(descriptors):
            if descriptors is None:
                return np.zeros(self.num_clusters)
            clusters = self.kmeans.predict(descriptors)
            histogram, _ = np.histogram(clusters, bins=np.arange(self.num_clusters + 1))
            return histogram

        hist = X.apply(create_histogram)
        return np.vstack(hist.to_numpy())
    

    @staticmethod
    def find_best_cluster_number(df, feature_method='ORB', cluster_range=range(100, 1001, 100)):
        """
        Finds the best cluster number using the Elbow Method based on inertia.

        Parameters:
        df (pd.DataFrame): A pandas DataFrame containing images.
        feature_method (str): The feature extraction method ('ORB' or 'SIFT').
        cluster_range (range): A range of cluster numbers to evaluate.

        Returns:
        pd.DataFrame: A DataFrame containing cluster numbers and their corresponding inertia values.
        """
        def extract_images(df):
            return df['image']

        # Define the pipeline for image extraction and feature extraction
        pipeline = Pipeline([
            ('extract_images', FunctionTransformer(extract_images, validate=False)),
            ('feature_extractor', FeatureExtractor(method=feature_method))
        ])

        # Apply the pipeline to the DataFrame to extract features
        features = pipeline.fit_transform(df)

        # Stack the features into a single NumPy array
        features = np.vstack(features.to_numpy())

        # Initialize lists to store cluster numbers and inertia values
        cluster_numbers = []
        inertia_values = []
        computational_times = []


        # Perform KMeans clustering for each cluster number and calculate inertia
        for k in cluster_range:
            start_time = time.time()
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            end_time = time.time()
            inertia = kmeans.inertia_
            computational_time = end_time - start_time
            cluster_numbers.append(k)
            inertia_values.append(inertia)
            computational_times.append(computational_time)
            print(f"Cluster number: {k}, Inertia: {inertia}, Time: {computational_time:.2f} seconds")

        # Create a DataFrame containing cluster numbers, inertia values, and computational time
        inertia_df = pd.DataFrame({
            'Cluster Number': cluster_numbers,
            'Inertia': inertia_values,
            'Computational Time (s)': computational_times
        })

        # Plot the inertia values to visualize the Elbow Method
        plt.figure(figsize=(8, 6))
        plt.plot(cluster_numbers, inertia_values, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()

        # Find the best cluster number based on the Elbow Method
        best_cluster_number = cluster_numbers[np.argmin(np.diff(inertia_values, 2))]
        print(f"Best cluster number based on the Elbow Method: {best_cluster_number}")

        return inertia_df
