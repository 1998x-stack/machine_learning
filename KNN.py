import numpy as np
from collections import Counter
from typing import List, Tuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

class KNNClassifier:
    def __init__(self, k: int = 3):
        """Initialize the classifier with a specified number of neighbors."""
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """Store training data and labels."""
        self.data = data
        self.labels = labels

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_neighbors(self, x: np.ndarray) -> List[Tuple[float, int]]:
        """Find the k nearest neighbors of the input point."""
        distances = [(self._euclidean_distance(x, self.data[i]), self.labels[i]) for i in range(len(self.data))]
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels for a set of test data points."""
        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            neighbor_classes = [neighbor[1] for neighbor in neighbors]
            most_common = Counter(neighbor_classes).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the accuracy of the classifier."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    # Visualization of decision boundaries (considering first two features only)
    @staticmethod
    def plot_decision_boundary(model, X_train, y_train, X_test, y_test):
        """Visualize the decision boundary by predicting labels for a grid of points."""
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        grid_points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size), np.zeros(xx.size)]
        predictions = model.predict(grid_points)
        predictions = predictions.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, predictions, alpha=0.5, levels=np.unique(y_train), cmap=plt.cm.Spectral)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='Train', edgecolors='k', cmap=plt.cm.Spectral, s=100)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', label='Test', edgecolors='k', cmap=plt.cm.Spectral, s=100)
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1])
        plt.legend()
        plt.title("KNN Decision Boundary (First 2 Features)")
        plt.savefig('figures/KNN_decision_boundary.png')
        plt.close()


if __name__ == '__main__':
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the KNN model
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)

    # Evaluate the model accuracy
    accuracy = knn.score(X_test, y_test)

    # Print the accuracy
    print(f"KNN Classifier Accuracy: {accuracy:.2f}")
    # Plot the decision boundary using only the first two features for visualization
    knn.plot_decision_boundary(knn, X_train[:, :2], y_train, X_test[:, :2], y_test)