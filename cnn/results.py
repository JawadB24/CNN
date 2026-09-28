import numpy as np
import matplotlib.pyplot as plt

class Results:
    """
    """

    def accuracy(self, A: np.ndarray, y: np.ndarray) -> float:
        """
        """

        predictions = np.argmax(A, axis=1)
        accuracy = np.mean(predictions == y) * 100

        return accuracy

    def confusion_matrix(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        """

        predictions = np.argmax(A, axis=1)
        classes = A.shape[1]
        cm = np.zeros((classes, classes))

        np.add.at(cm, (y, predictions), 1)

        return cm

    def learning_curve(self, cost_history: list[float]) -> None:
        """
        """

        plt.plot(cost_history)
        plt.title("Learning Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.show()