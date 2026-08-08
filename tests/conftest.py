import pytest
import numpy as np
from sklearn.datasets import make_regression


@pytest.fixture(scope="module")
def mock_regression_data_1D():
    """
    Creates a synthetic regression dataset.
    scope="module" means this data is created once per test file,
    saving computation time.
    """
    bias = 1.2
    X,y,coef = make_regression(
        n_samples=500,
        n_features=1,
        noise=10,
        random_state=42,
        coef=True,
        bias=bias
    )

    return X.flatten(), y , coef, bias

@pytest.fixture(scope="module")
def mock_regression_data_ND():
    """
    Creates a synthetic regression dataset.
    scope="module" means this data is created once per test file,
    saving computation time.
    """
    bias = 1.2
    X,y,coef = make_regression(
        n_samples=500,
        n_features=10,
        noise=5,
        random_state=42,
        coef=True,
        bias=bias
    )

    return X, y , coef, bias



