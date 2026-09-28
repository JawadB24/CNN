import numpy as np
from cnn.neuralnet import NeuralNet

class Optimizer:
    """
    """

    def adams_optimization(self, epochs: int, alpha: float, batch_size: int, NN: NeuralNet, X: np.ndarray, y: np.ndarray) -> list[float]:
        """
        """

        cost_history = []
        m = X.shape[0]
        t = 1

        for epoch in range(epochs):

            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            epoch_cost = 0.0

            for start in range(0, m, batch_size):
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                A_batch = NN.forward(X_batch)
                NN.backward(A_batch, y_batch)
                NN.update(alpha, t)
                t += 1

                batch_cost = NN.cost_function(A_batch, y_batch)
                epoch_cost += (batch_cost * X_batch.shape[0])

            avg_epoch_cost = epoch_cost / m
            cost_history.append(avg_epoch_cost)
            print(f"Average Cost (Epoch {epoch}) : {avg_epoch_cost}")
        
        return cost_history
