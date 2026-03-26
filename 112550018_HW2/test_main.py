import numpy as np
import pytest
from loguru import logger
from sklearn.metrics import accuracy_score

from main import FLD, LogisticRegression


def generate_pseudo_data(num_samples, slope, intercept):
    np.random.seed(42)
    X = np.random.rand(num_samples, 2) * 10  # Generate random points between 0 and 10
    y = (X[:, 1] > slope * X[:, 0] + intercept).astype(int)
    index = np.logical_or(X[:, 1] <= 4, X[:, 1] >= 6)
    X, y = X[index], y[index]
    print(X.shape, y.shape)
    return X, y


@pytest.fixture
def sample_data():
    slope = 0.5
    intercept = 2
    num_samples = 500
    X, y = generate_pseudo_data(num_samples, slope, intercept)
    return X, y


def test_logistic_regression(sample_data):
    inputs, targets = sample_data  # (n, 2), (n, )
    model = LogisticRegression(learning_rate=1e-3, num_iterations=30000)
    model.fit(inputs[:250], targets[:250])

    y_pred_probs, y_pred_classes = model.predict(inputs[250:])
    accuracy = accuracy_score(targets[250:], y_pred_classes)
    logger.info(f'{accuracy=:.4f}')
    assert accuracy >= 0.9


def test_fld(sample_data):
    inputs, targets = sample_data
    model = FLD()
    model.fit(inputs[:250], targets[:250])
    y_preds = model.predict(inputs[250:])
    accuracy = accuracy_score(targets[250:], y_preds)
    logger.info(f'{accuracy=:.4f}')
    model.plot_projection(inputs[250:], targets[250:], y_preds)
    assert accuracy >= 0.8
