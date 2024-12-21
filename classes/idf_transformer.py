import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IDFTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        N = len(X)
        df_document_frequency = np.sum(np.vstack(X) > 0, axis=0)
        self.idf = np.log((N + 1) / (df_document_frequency + 1)) + 1
        return self

    def transform(self, X):
        return X.apply(lambda hist: hist * self.idf)