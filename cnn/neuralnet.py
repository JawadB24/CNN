import numpy as np
from cnn.layers import Layer, Conv, BatchNorm, ReLU, Pooling, Flatten, Dense, Softmax

class NeuralNet:
    """
    """

    def __init__(self) -> None:
        self.layers = []


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        """

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def prediction_gradient(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        """

        output_layer = self.layers[-1]
        d_output = output_layer.backward(A, y)

        return d_output

    def backward(self, A: np.ndarray, y: np.ndarray) -> None:
        """
        """

        d_output = self.prediction_gradient(A, y)

        for layer in reversed(self.layers[:-1]):
            d_output = layer.backward(d_output)


    def cost_function(self, A: np.ndarray, y: np.ndarray) -> None:
        """
        """

        m = A.shape[0]
        cost = -1 * np.mean(np.log(A[np.arange(m), y]))

        return cost

    def update(self, alpha: float, t: int) -> np.ndarray:
        """
        """

        for layer in self.layers:
            layer.update(alpha, t)

    def set_dtype(self, dtype) -> None:
        """
        """

        for layer in self.layers:

            if hasattr(layer, "W"):
                layer.W = layer.W.astype(dtype)

            if hasattr(layer, "b"):
                layer.b = layer.b.astype(dtype)

            if hasattr(layer, "gamma"):
                layer.gamma = layer.gamma.astype(dtype)

            if hasattr(layer, "beta"):
                layer.beta = layer.beta.astype(dtype)

    
    def gradient_checking(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        """

        epsilon = 10 ** (-1 * 5)
        learnable_layers = 0
        avg_error = 0

        A = self.forward(X)
        self.backward(A, y)

        for layer in self.layers:

            dtheta_approx = []
            dtheta = []

            if hasattr(layer, "W"):
                for index in np.ndindex(layer.W.shape):
                    original = layer.W[index]

                    layer.W[index] = original + epsilon

                    A_right = self.forward(X)
                    cost_right = self.cost_function(A_right, y)

                    layer.W[index] = original - epsilon

                    A_left = self.forward(X)
                    cost_left = self.cost_function(A_left, y)

                    gradient_approx = (cost_right - cost_left) / (2 * epsilon)
                    gradient = layer.dW[index]

                    dtheta_approx.append(gradient_approx)
                    dtheta.append(gradient)

                    layer.W[index] = original
                
            if hasattr(layer, "b"):
                for index in np.ndindex(layer.b.shape):
                    original = layer.b[index]

                    layer.b[index] = original + epsilon

                    A_right = self.forward(X)
                    cost_right = self.cost_function(A_right, y)

                    layer.b[index] = original - epsilon

                    A_left = self.forward(X)
                    cost_left = self.cost_function(A_left, y)

                    gradient_approx = (cost_right - cost_left) / (2 * epsilon)
                    gradient = layer.db[index]

                    dtheta_approx.append(gradient_approx)
                    dtheta.append(gradient)

                    layer.b[index] = original
                
            if hasattr(layer, "gamma"):
                for index in np.ndindex(layer.gamma.shape):
                    original = layer.gamma[index]

                    layer.gamma[index] = original + epsilon

                    A_right = self.forward(X)
                    cost_right = self.cost_function(A_right, y)

                    layer.gamma[index] = original - epsilon

                    A_left = self.forward(X)
                    cost_left = self.cost_function(A_left, y)

                    gradient_approx = (cost_right - cost_left) / (2 * epsilon)
                    gradient = layer.dgamma[index]

                    dtheta_approx.append(gradient_approx)
                    dtheta.append(gradient)

                    layer.gamma[index] = original
                
            if hasattr(layer, "beta"):
                for index in np.ndindex(layer.beta.shape):
                    original = layer.beta[index]

                    layer.beta[index] = original + epsilon

                    A_right = self.forward(X)
                    cost_right = self.cost_function(A_right, y)

                    layer.beta[index] = original - epsilon

                    A_left = self.forward(X)
                    cost_left = self.cost_function(A_left, y)

                    gradient_approx = (cost_right - cost_left) / (2 * epsilon)
                    gradient = layer.dbeta[index]

                    dtheta_approx.append(gradient_approx)
                    dtheta.append(gradient)

                    layer.beta[index] = original

            if dtheta:
                dtheta_approx = np.array(dtheta_approx)
                dtheta = np.array(dtheta)

                error = np.linalg.norm(dtheta_approx - dtheta) / (np.linalg.norm(dtheta_approx) + np.linalg.norm(dtheta))
                avg_error += error
                learnable_layers += 1

                print(f"{type(layer).__name__} Relative Error: {error}")

        print(f"Average Relative Error : {avg_error / learnable_layers}")


