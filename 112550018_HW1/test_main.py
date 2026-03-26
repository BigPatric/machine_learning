"""Question5: PyTest.
Don't forget to fix the input args to align your implementation.
"""
import numpy as np
import pytest
from loguru import logger
from main import LinearRegressionCloseform, LinearRegressionGradientdescent

@pytest.fixture
def slope_and_intercept():
    slope = 3
    intercept = 4
    return slope, intercept


@pytest.fixture
def sample_data(slope_and_intercept):
    slope, intercept = slope_and_intercept
    n_datapoints = 100
    xs = np.linspace(-100, 100, n_datapoints).reshape((n_datapoints, 1))
    print(xs.shape)
    ys = slope * xs + intercept
    return xs, ys


def test_regression_cf(sample_data, slope_and_intercept):
    x, y = sample_data
    model = LinearRegressionCloseform()
    model.fit(x, y)
    logger.info(f'{model.weights=}, {model.intercept=}')

    slope, intercept = slope_and_intercept

    """Don't modify the assertion"""
    assert model.weights[0] == pytest.approx(slope, 0.1)
    assert model.intercept == pytest.approx(intercept, 0.1)


def test_regression_gd(sample_data, slope_and_intercept):
    x, y = sample_data
    model = LinearRegressionGradientdescent()
    model.fit(x, y, learning_rate=2e-4, epochs=100000)

    logger.info(f'{model.weights=}, {model.intercept=}')
    slope, intercept = slope_and_intercept

    """Don't modify the assertion"""
    assert model.weights[0] == pytest.approx(slope, 0.1)
    assert model.intercept == pytest.approx(intercept, 0.1)
