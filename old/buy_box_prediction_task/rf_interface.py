# rf_interface.py

import joblib
import numpy as np

PIPELINE_PATH = 'pipe_red_rf.joblib'

def load_pipeline(path=PIPELINE_PATH):
    """
    Load the saved sklearn Pipeline from disk.

    Args:
        path (str): Path to the .joblib file.

    Returns:
        pipeline (sklearn.Pipeline): The loaded model pipeline.
    """
    pipeline = joblib.load(path)
    print(f"Loaded pipeline from '{path}'")
    return pipeline


def predict(pipeline, feature_vector):
    """
    Run prediction on a single feature vector.

    Args:
        pipeline (sklearn.Pipeline): The trained pipeline.
        feature_vector (array-like): 1D array of length n_features.

    Returns:
        pred (int): Predicted class label (0 or 1).
        prob (float): Predicted probability for class=1.
    """
    # Convert to 2D array: shape (1, n_features)
    X = np.array(feature_vector).reshape(1, -1)
    pred = pipeline.predict(X)[0]
    prob = pipeline.predict_proba(X)[0, 1]
    return pred, prob
