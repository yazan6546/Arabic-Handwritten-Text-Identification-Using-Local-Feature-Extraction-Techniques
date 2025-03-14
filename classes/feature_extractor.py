import cv2
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, method='ORB'):
        self.method = method
        if method == 'ORB':
            self.detector = cv2.ORB_create(
                                nfeatures=750,
                                scaleFactor=1.2,
                                nlevels=8,
                                edgeThreshold=15,
                                firstLevel=0,
                                WTA_K=2,
                                scoreType=cv2.ORB_HARRIS_SCORE,
                                patchSize=15,
                                fastThreshold=10
                            )
            
        elif method == 'SIFT':
            self.detector = cv2.SIFT_create(
                                nfeatures=750,
                                nOctaveLayers=3,
                                contrastThreshold=0.01,
                                edgeThreshold=15,
                                sigma=1.2
                            )   
        else:
            raise ValueError("Unsupported method")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def extract(image):
            _, descriptors = self.detector.detectAndCompute(image, None)
            return descriptors

        return X.apply(extract)

    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the cv2.ORB or cv2.SIFT object from the state
        state['detector'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the cv2.ORB or cv2.SIFT object
        if self.method == 'ORB':
            self.detector = cv2.ORB_create(
                                nfeatures=750,
                                scaleFactor=1.2,
                                nlevels=8,
                                edgeThreshold=15,
                                firstLevel=0,
                                WTA_K=2,
                                scoreType=cv2.ORB_HARRIS_SCORE,
                                patchSize=15,
                                fastThreshold=10
                            )
        elif self.method == 'SIFT':
            self.detector = cv2.SIFT_create(
                                nfeatures=750,
                                nOctaveLayers=3,
                                contrastThreshold=0.01,
                                edgeThreshold=15,
                                sigma=1.2
                            )  