import warnings

import numpy as np


class RidgeRegressionGD:
    """
    Implements Multiple Ridge Regression using NumPy.

    This class provides methods to fit a multiple ridge regression model
    to training data and make predictions on new data. It calculates the
    coefficients and intercept using Gradient descent.
    """
    def __init__(self,alpha:float = 0.1,learning_rate:float = 0.001,epochs:int = 500):
        self.intercept_ = None
        self.coef_ = None
        if epochs <= 0:
            warnings.warn("The epochs passed was negative or zero.Defaulting to 500.",UserWarning)
            epochs = 500 
        self.epochs = epochs
        if alpha < 0:
            warnings.warn("The alpha passed was negative.Treating it as positive.",UserWarning)
            alpha = abs(alpha)
        self.alpha = alpha
        if learning_rate < 0:
            warnings.warn("The learning passed was negative.Treating it as positive.",UserWarning)
            learning_rate = abs(learning_rate)
        self.learning_rate = learning_rate
        
    def fit(self,X_train: np.ndarray, y_train:np.ndarray) -> None:
        """
         Attributes:
            coef_ (np.ndarray): The learned coefficients for the features.
            intercept_ (float): The learned intercept (bias) term.
        """
        # Handle 1D input (single feature case)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            
        # Initialise all the W to 1
        self.coef_ = np.ones(X_train.shape[1])
        self.intercept_ = 0
        
        # create the W matrix aliased as theta 
        theta = np.insert(self.coef_,0,self.intercept_)
        
        X_train = np.insert(X_train,0,1,axis=1)
        # Add Gradient Scaling
        m = X_train.shape[0]
        
        for i in range(self.epochs):
            theta_der = (1/m) * (np.dot(X_train.T,X_train).dot(theta) - np.dot(X_train.T,y_train) + self.alpha * np.insert(theta[1:],0,0))
            theta = theta - self.learning_rate * theta_der
            
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self,X_test: np.ndarray) -> np.ndarray:
        """
         Args:
            X_train (np.ndarray): Training feature data. Can be 1D or 2D.
                                  If 1D, it will be reshaped to 2D.
            y_train (np.ndarray): Training target values.
        """
        # Handle 1D input
        is_1D = False
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
            is_1D = True
        
        # Handle 1D coefficients
        if self.coef_.ndim == 1:
            self.coef_ = self.coef_.reshape(-1, 1)
        
        y_pred = np.dot(X_test, self.coef_) + self.intercept_
        
        # Flatten if input was 1D
        if is_1D:
            y_pred = y_pred.flatten()
        
        return y_pred
        
        