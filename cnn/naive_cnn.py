import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
class Layer(ABC):
    """
    """

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        """
        pass

    @abstractmethod
    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        """
        pass

    @abstractmethod
    def update(self, alpha: float) -> None:
        """
        """
        pass


class Conv(Layer):
    """
    """

    def __init__(self, n_filters: int, n_c: int, f_h: int, f_w: int, stride: int, padding: int) -> None:
        """
        """

        self.X_padded = None
        self.X_2D = None
        self.W = self.kernel_initialization(n_filters, n_c, f_h, f_w)
        #self.b = np.zeros((1, n_filters, 1, 1)).astype(np.float32)
        self.dW = None
        self.VdW = np.zeros_like(self.W)
        self.SdW = np.zeros_like(self.W)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10 ** (-1 * 8)
        #self.Vdb = None
        #self.Sdb = None
        #self.db = None
        self.n_filters = n_filters
        self.n_c = n_c
        self.f_h = f_h
        self.f_w = f_w
        self.n_h_prime = None
        self.n_w_prime = None
        self.stride = stride
        self.padding = padding

    def kernel_initialization(self, n_filters: int, n_c: int, f_h: int, f_w: int) -> np.ndarray:
        """
        """

        var = 2 / (n_c * f_h * f_w)
        std = np.sqrt(var)
        mean = 0

        W = np.random.normal(loc=mean, scale=std, size=(n_filters, n_c, f_h, f_w)).astype((np.float32))

        return W

    def cross_correlation(self, X: np.ndarray, W: np.ndarray, stride: int, padding: int) -> np.ndarray:
        """
        """

        X_padded = np.pad(X, pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
        self.X_padded = X_padded

        m = X.shape[0]
        #n_c = X.shape[1] = self.n_c
        n_h = X.shape[2]
        n_w = X.shape[3]

        n_h_prime = ((n_h + 2 * padding - self.f_h) // stride) + 1
        n_w_prime = ((n_w + 2 * padding - self.f_w) // stride) + 1

        Z = np.zeros((m, self.n_filters, n_h_prime, n_w_prime)).astype(np.float32)

        for i in range(m):
            for k in range(self.n_filters):
                for c in range(self.n_c):
                    for h_prime in range(n_h_prime):
                        for w_prime in range(n_w_prime):
                            for j_h in range(self.f_h):
                                for j_w in range(self.f_w):
                                    Z[i][k][h_prime][w_prime] += (W[k][c][j_h][j_w] * self.X_padded[i][c][h_prime * stride + j_h][w_prime * stride + j_w])
        return Z

    def parameter_gradient(self, X: np.ndarray, W: np.ndarray, dZ: np.ndarray) -> None:
        """
        """

        #self.db = np.sum(dZ, axis=(0, 2, 3), keepdims=True) each bias is associated with each filter, which runs over all examples

        m = X.shape[0]
        #n_c = X.shape[1] = self.n_c

        #self.n_filters = dZ.shape[1]
        n_h_prime = dZ.shape[2]
        n_w_prime = dZ.shape[3]

        dW = np.zeros_like(W, dtype=W.dtype)

        for k in range(self.n_filters): #Intuitively, iterating over each image for a given filter makes sense for weight gradient computation,
                                        #however the order in which they are iterated does not matter
            for i in range(m):
                for c in range(self.n_c):
                    for j_h in range(self.f_h):
                        for j_w in range(self.f_w):
                            for h_prime in range(n_h_prime):
                                for w_prime in range(n_w_prime):
                                    dW[k][c][j_h][j_w] += (dZ[i][k][h_prime][w_prime] * self.X_padded[i][c][h_prime * self.stride + j_h][w_prime * self.stride + j_w])
        self.dW = dW

    def input_gradient(self, X: np.ndarray, W: np.ndarray, dZ: np.ndarray) -> np.ndarray:
        """
        """

        m = X.shape[0]

        n_h_prime = dZ.shape[2]
        n_w_prime = dZ.shape[3]

        dX = np.zeros_like(X, dtype=X.dtype)

        for i in range(m):
            for k in range(self.n_filters):
                for c in range(self.n_c):
                    for h_prime in range(n_h_prime):
                        for w_prime in range(n_w_prime):
                            for j_h in range(self.f_h):
                                for j_w in range(self.f_w):
                                    h = h_prime * self.stride + j_h 
                                    w = w_prime * self.stride + j_w
                                    #all values of h and w using weight indices and dZ indices, thus the derivative is as below
                                    dX[i][c][h][w] += (W[k][c][j_h][j_w] * dZ[i][k][h_prime][w_prime])
        return dX

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        """

        Z = self.cross_correlation(X, self.W, self.stride, self.padding) #+ self.b

        return Z

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        """

        self.parameter_gradient(self.X_padded, self.W, dZ)
        dX_padded = self.input_gradient(self.X_padded, self.W, dZ)
        n_h_p, n_w_p = dX_padded.shape[2], dX_padded.shape[3]
        dX = dX_padded[:, :, self.padding:n_h_p - self.padding, self.padding: n_w_p - self.padding]

        return dX

    def update(self, alpha: float, t: int) -> None:
        """
        """

        self.VdW = self.beta1 * self.VdW + (1 - self.beta1) * self.dW
        self.SdW = self.beta2 * self.SdW + (1 - self.beta2) * (self.dW ** 2)

        #self.Vdb = self.beta1 * self.Vdb + (1 - self.beta1) * self.db
        #self.Sdb = self.beta2 * self.Sdb + (1 - self.beta2) * (self.db ** 2)

        VdW_corrected = self.VdW / (1 - self.beta1 ** t)
        SdW_corrected = self.SdW / (1 - self.beta2 ** t)


        #Vdb_corrected = self.Vdb / (1 - self.beta1 ** t)
        #Sdb_corrected = self.Sdb / (1 - self.beta2 ** t)

        self.W -= alpha * (VdW_corrected / np.sqrt(SdW_corrected + self.epsilon))

        #self.b -= alpha * (Vdb_corrected / np.sqrt(Sdb_corrected + self.epsilon))

class BatchNorm(Layer):
    """
    """

    def __init__(self, n_filters: int, input_ndim: int) -> None:
        """
        """

        self.Z_norm = None
        self.n_filters = n_filters
        self.input_ndim = input_ndim

        if input_ndim == 4:
            self.gamma = np.ones((1, self.n_filters, 1, 1), dtype=np.float32)
            self.beta = np.zeros((1, self.n_filters, 1, 1), dtype=np.float32)
        else:
            self.gamma = np.ones((1, self.n_filters), dtype=np.float32)
            self.beta = np.zeros((1, self.n_filters), dtype=np.float32)

        self.dgamma = None
        self.dbeta = None
        self.Vdgamma = np.zeros_like(self.gamma)
        self.Sdgamma = np.zeros_like(self.gamma)
        self.Vdbeta = np.zeros_like(self.beta)
        self.Sdbeta = np.zeros_like(self.beta)
        self.var = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10 ** (-1 * 8)


    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        """

        if self.input_ndim == 4:
            axes = (0, 2, 3)
        else:
            axes = 0

        mean = np.mean(Z, axis=axes, keepdims=True)
        self.var = np.var(Z, axis=axes, keepdims=True)

        self.Z_norm = (Z - mean) / np.sqrt(self.var + self.epsilon)

        Z_tilde = self.gamma * self.Z_norm + self.beta

        return Z_tilde

    def backward(self, dZ_tilde: np.ndarray) -> np.ndarray:
        """
        """

        if self.input_ndim == 4:
            axes = (0, 2, 3)
        else:
            axes = 0

        self.dgamma = np.sum(dZ_tilde * self.Z_norm, axis=axes, keepdims=True)
        self.dbeta = np.sum(dZ_tilde, axis=axes, keepdims=True)

        dZ = (self.gamma / np.sqrt(self.var + self.epsilon)) * (dZ_tilde - np.mean(dZ_tilde, axis=axes, keepdims=True) - self.Z_norm * np.mean(dZ_tilde * self.Z_norm, axis=axes, keepdims=True))

        return dZ

    def update(self, alpha: float, t: int) -> None:
        """
        """
        self.Vdgamma = self.beta1 * self.Vdgamma + (1 - self.beta1) * self.dgamma
        self.Sdgamma = self.beta2 * self.Sdgamma + (1 - self.beta2) * (self.dgamma ** 2)

        self.Vdbeta = self.beta1 * self.Vdbeta + (1 - self.beta1) * self.dbeta
        self.Sdbeta = self.beta2 * self.Sdbeta + (1 - self.beta2) * (self.dbeta ** 2)

        Vdgamma_corrected = self.Vdgamma / (1 - self.beta1 ** t)
        Sdgamma_corrected = self.Sdgamma / (1 - self.beta2 ** t)

        Vdbeta_corrected = self.Vdbeta / (1 - self.beta1 ** t)
        Sdbeta_corrected = self.Sdbeta / (1 - self.beta2 ** t)

        self.gamma -= alpha * (Vdgamma_corrected / np.sqrt(Sdgamma_corrected + self.epsilon))
        self.beta -= alpha * (Vdbeta_corrected / np.sqrt(Sdbeta_corrected + self.epsilon))


class ReLU(Layer):
    """
    """

    def __init__(self) -> None:
        """
        """

        self.Z_tilde = None


    def forward(self, Z_tilde: np.ndarray) -> None:
        """
        """

        self.Z_tilde = Z_tilde
        A = np.maximum(self.Z_tilde, 0)

        return A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        """

        dZ_tilde = dA * (self.Z_tilde > 0)

        return dZ_tilde

    def update(self, alpha: float, t: float) -> np.ndarray:
        """
        """
        pass

class Pooling(Layer):
    """
    """

    def __init__(self, p_h: int, p_w: int, stride: int) -> None:
        """
        """
        self.A = None
        self.p_h = p_h
        self.p_w = p_w
        self.stride = stride
        self.indices = []

    def max_pooling(self, A: np.ndarray, p_h: int, p_w: int, stride: int) -> np.ndarray:
        """
        """

        m = A.shape[0]
        n_filters = A.shape[1] #the number of channels in the output of a conv layer is equal to the number of filters in the conv layer
        n_h_prime = A.shape[2]
        n_w_prime = A.shape[3]


        mp_h = ((n_h_prime - p_h) // stride) + 1
        mp_w = ((n_w_prime - p_w) // stride) + 1

        max_pool = np.empty((m, n_filters, mp_h, mp_w), dtype=A.dtype)

        for i in range(m):
            for k in range(n_filters):
                for m_h in range(mp_h):
                    for m_w in range(mp_w):
                                window = A[i][k][m_h * stride:m_h * stride + p_h, m_w * stride:m_w * stride + p_w]
                                max_value = np.max(window)
                                window_index = np.unravel_index(np.argmax(window), window.shape) #argmax flattens the window, extracts index, 
                                #and unravel_index converts the flattened index back into its multidimensional coordinates
                                max_h = m_h * stride + window_index[0]
                                max_w = m_w * stride + window_index[1]
                                self.indices.append([max_h, max_w])
                                max_pool[i][k][m_h][m_w] = max_value
        return max_pool

    def forward(self, A: np.ndarray) -> np.ndarray:
        """
        """
        self.indices = []
        self.A = A
        A_pooled = self.max_pooling(self.A, self.p_h, self.p_w, self.stride)

        return A_pooled

    def backward(self, dA_pooled: np.ndarray) -> np.ndarray:
        """
        """

        m = dA_pooled.shape[0]
        n_filters = dA_pooled.shape[1]
        mp_h = dA_pooled.shape[2]
        mp_w = dA_pooled.shape[3]

        dA = np.zeros_like(self.A, dtype=self.A.dtype)
        counter = 0

        for i in range(m):
            for k in range(n_filters):
                for m_h in range(mp_h):
                    for m_w in range(mp_w):
                        max_h = self.indices[counter][0]
                        max_w = self.indices[counter][1]
                        dA[i][k][max_h][max_w] += dA_pooled[i][k][m_h][m_w]
                        counter += 1    

        return dA

    def update(self, alpha: float, t: float) -> np.ndarray:
        """
        """
        pass


class Flatten(Layer):
    """
    """

    def __init__(self) -> None:
        """
        """
        self.X_shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        """

        self.X_shape = X.shape
        flattened_input = X.reshape(X.shape[0], -1)

        return flattened_input

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        """

        dZ_fitted = dZ.reshape(self.X_shape)

        return dZ_fitted

    def update(self, alpha: float, t: int) -> None:
        """
        """
        pass

class Dense(Layer):
    """
    """

    def __init__(self, fan_in: int, fan_out: int) -> np.ndarray:
        """
        """

        self.X = None
        self.W = self.weight_initialization(fan_in, fan_out)
        self.b = np.zeros((1, fan_out), dtype=np.float32)
        self.dW = None
        self.VdW = np.zeros_like(self.W)
        self.SdW = np.zeros_like(self.W)
        self.db = None
        self.Vdb = np.zeros_like(self.b)
        self.Sdb = np.zeros_like(self.b)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10 ** (-1 * 8)
        self.fan_in = fan_in
        self.fan_out = fan_out

    def weight_initialization(self, fan_in: int, fan_out: int) -> np.ndarray:
        """
        """

        var = 2 / fan_in
        std = np.sqrt(var)
        mean = 0

        W = np.random.normal(loc=mean, scale=std, size=(fan_in, fan_out)).astype(np.float32)

        return W

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        """

        self.X = X

        Z = self.X @ self.W + self.b

        return Z

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        """

        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        dX = dZ @ self.W.T

        return dX

    def update(self, alpha: float, t: int):
        """
        """

        self.VdW = self.beta1 * self.VdW + (1 - self.beta1) * self.dW
        self.SdW = self.beta2 * self.SdW + (1 - self.beta2) * (self.dW ** 2)

        self.Vdb = self.beta1 * self.Vdb + (1 - self.beta1) * self.db
        self.Sdb = self.beta2 * self.Sdb + (1 - self.beta2) * (self.db ** 2)

        VdW_corrected = self.VdW / (1 - self.beta1 ** t)
        SdW_corrected = self.SdW / (1 - self.beta2 ** t)


        Vdb_corrected = self.Vdb / (1 - self.beta1 ** t)
        Sdb_corrected = self.Sdb / (1 - self.beta2 ** t)

        self.W -= alpha * (VdW_corrected / np.sqrt(SdW_corrected + self.epsilon))

        self.b -= alpha * (Vdb_corrected / np.sqrt(Sdb_corrected + self.epsilon))


class Softmax(Layer):
    """
    """

    def __init__(self) -> np.ndarray:
        """
        """

        self.Z = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        """

        self.Z = Z
        exp_Z = np.exp(self.Z)
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        return A

    def backward(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        """

        m = A.shape[0]
        dZ = A.copy()

        dZ[np.arange(m), y] -= 1
        dZ = (1 / m) * dZ

        return dZ

    def update(self, alpha: float, t: float) -> None:
        """
        """
        pass

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
    
    def gradient_checking(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        """

        epsilon = 10 ** (-1 * 5)

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

                print(f"{type(layer).__name__}: {error}")

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
                        











                                                                  

