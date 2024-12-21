import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

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

        return X.apply(create_histogram)