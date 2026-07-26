import numpy as np

class MultipleRidgeRegression:
    """
    Implements Multiple Ridge Regression using NumPy.

    This class provides methods to fit a multiple ridge regression model
    to training data and make predictions on new data. It calculates the
    coefficients and intercept using the closed-form solution (Normal Equation).
    """
    def __init__(self,alpha = 0.1):
        """
        Initializes the MultipleLinearRegression model.

        Attributes:
            coef_ (np.ndarray): The learned coefficients for the features.
            intercept_ (float): The learned intercept (bias) term.
            alpha: The regularisation term.
        """
        self.alpha = alpha
        self.coef_  = None
        self.intercept_ = None

    def fit(self,X_train:np.ndarray,y_train:np.ndarray):
        """
        Fits the multiple ridge regression model to the training data.

        The coefficients and intercept are calculated using the Normal Equation:
        betas = (X_train.T @ X_train)^-1 @ X_train.T @ y_train

        Args:
            X_train (np.ndarray): Training feature data. Can be 1D or 2D.
                                  If 1D, it will be reshaped to 2D.
            y_train (np.ndarray): Training target values.
        """
         # Ensure that the array is 2D even if the array is 1D
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        # Add a column of ones to X_train for the intercept term
        X_train = np.insert(X_train,0,1,axis = 1)
        # Add the identity matrix
        I = np.identity(X_train.shape[1])
        # Set the first row and column's value to zero.
        I[0][0] = 0
        
        
        # Calculate betas (coefficients and intercept) using the Normal Equation
        # np.linalg.inv: computes the inverse of a matrix
        # np.dot: performs dot product of two arrays
        betas = np.linalg.inv(np.dot(X_train.T,X_train) + self.alpha * I).dot(X_train.T).dot(y_train)
        
        # get the intercept from the matrix
        self.intercept_ = betas[0] 
        self.coef_ = betas[1:]

    def predict(self,X_test:np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained model.

        Args:
            X_test (np.ndarray): Feature data for making predictions. Can be 1D or 2D.
                                 If 1D, it will be reshaped to 2D.

        Returns:
            np.ndarray: Predicted target values.
        """
        is_1D = False 
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1,1)
            # Set the flag to True if the input array is 1D
            is_1D = True 
        if self.coef_.ndim == 1:
            self.coef_ = self.coef_.reshape(-1,1)
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        if is_1D:
            # Flatten the array so as to match the original array
            y_pred = y_pred.flatten() 
        return y_pred 
        