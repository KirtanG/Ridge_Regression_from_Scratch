import numpy as np

class SimpleRidgeRegression:
    
    def __init__(self,alpha = 0.1) -> None:
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha

    def fit(self,X_train: np.ndarray, y_train: np.ndarray) -> None:

        if X_train.ndim != 1:
            raise RuntimeError("This implementation supports 1D features only.")
        
        X_train = X_train.ravel()

        X_mean = X_train.mean()
        y_mean = y_train.mean()

        numerator = np.sum((X_train - X_mean) * (y_train - y_mean))
        denominator = np.sum((X_train - X_mean)**2)

        self.coef_ = numerator/(denominator + self.alpha)

        self.intercept_ = y_mean - (self.coef_ * X_mean)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return (X_test * self.coef_) + self.intercept_