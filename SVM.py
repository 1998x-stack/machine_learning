import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
from ml_dataset import load_data

class SVM:
    """
    Support Vector Machine (SVM) implementation using NumPy.
    
    Attributes:
        learning_rate (float): Learning rate for weight updates.
        regularization (float): Regularization parameter for soft margin.
        epochs (int): Number of epochs for training.
        weights (np.ndarray): Weights of the SVM model.
        bias (float): Bias term.
    """
    def __init__(self, learning_rate: float = 0.001, regularization: float = 0.01, epochs: int = 1000):
        """
        Initialize the SVM model with given hyperparameters.
        
        Args:
            learning_rate (float): Learning rate for weight updates.
            regularization (float): Regularization parameter for soft margin.
            epochs (int): Number of epochs for training.
        """
        
        self.learning_rate = learning_rate # Learning rate for weight updates
        self.regularization = regularization # Regularization parameter for soft margin
        self.epochs = epochs # Number of epochs for training
        self.weights = None # Weights of the SVM model
        self.bias = None # Bias term
        
    def fit(self, X: np.ndarray, y: np.ndarray, kwargs) -> None:
        """
        Train the SVM model using the given data.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # Initialize weights to zeros
        self.bias = 0.0
        
        # convert labels to {-1, 1}
        y = np.where(y == kwargs['negative_labels'], -1, 1) # y==0 -> -1, y==1 -> 1
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Decision boundary condition
                condition = (y[idx] * (np.dot(x_i, self.weights) + self.bias)) >= 1
                
                if condition:
                    # Correct classification: only apply regularization
                    self.weights = self.weights - self.learning_rate * (2 * self.regularization * self.weights)
                else:
                    # Misclassification: apply hinge loss gradient
                    self.weights -= self.learning_rate * (2 * self.regularization * self.weights - np.dot(y[idx], x_i))
                    self.bias -= self.learning_rate * y[idx]
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data.
        
        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Predicted labels of shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)
                    
if __name__ == '__main__':
    # Load the Iris dataset (binary classification: Setosa vs. Versicolor)
    iris = load_data('iris')
    X = iris.data[iris.target != 2, :2]  # Only take first 2 features and 2 classes
    y = iris.target[iris.target != 2]

    # Initialize and fit the SVM model
    svm = SVM(learning_rate=0.001, regularization=3e-3, epochs=2000)
    svm.fit(X, y, {'negative_labels': 0})

    # Predict and visualize the decision boundary
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.01), np.arange(x1_min, x1_max, 0.01))
    Z = svm.predict(np.c_[xx0.ravel(), xx1.ravel()])
    Z = Z.reshape(xx0.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx0, xx1, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k', marker='o', label="Data Points")
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('SVM Decision Boundary on Iris Dataset')
    plt.grid(True)
    plt.savefig('figures/SVM.png')
    plt.close()