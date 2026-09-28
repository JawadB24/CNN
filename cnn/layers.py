import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        """
        pass

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
    def update(self, alpha: float, t: int) -> None:
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
        #self.b = np.zeros((1, n_filters, 1, 1))
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

    def im2col(self, X_padded: np.ndarray, n_c: int, f_h: int, f_w: int, stride: int) -> np.ndarray:
        """
        """

        m, n_h, n_w = X_padded.shape[0], X_padded.shape[2], X_padded.shape[3]

        self.n_h_prime = ((n_h - f_h) // stride) + 1
        self.n_w_prime = ((n_w - f_w) // stride) + 1

        X_windows = sliding_window_view(X_padded, (f_h, f_w), (2, 3)) #Think of as a 6D tensor (m x n_c x n_h_prime x n_w_prime x f_h x f_w)

        X_windows = X_windows[:, :, ::stride, ::stride, :, :] #skips over windows according to stride

        X_windows = X_windows.transpose(0, 2, 3, 1, 4, 5)

        self.X_2D = X_windows.reshape(m * self.n_h_prime * self.n_w_prime, n_c * f_h * f_w)

        return self.X_2D

    def cross_correlation(self, X: np.ndarray, W: np.ndarray, stride: int, padding: int) -> np.ndarray:
        """
        """

        m = X.shape[0]

        self.X_padded = np.pad(X, pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

        X_2D = self.im2col(self.X_padded, self.n_c, self.f_h, self.f_w, stride)

        W = W.reshape(self.n_filters, self.n_c * self.f_h * self.f_w)

        Z_2D = W @ X_2D.T

        Z = Z_2D.reshape(self.n_filters, m, self.n_h_prime, self.n_w_prime)
        Z = Z.transpose(1, 0, 2, 3)

        return Z


    def col2im(self, dX_2D: np.ndarray, n_c: int, f_h: int, f_w: int, stride: int) -> np.ndarray:
        """
        """

        m = self.X_padded.shape[0]

        dX = np.zeros_like(self.X_padded)

        dX_windows = dX_2D.reshape(m, self.n_h_prime, self.n_w_prime, n_c, f_h, f_w)
        dX_windows = dX_windows.transpose(0, 3, 1, 2, 4, 5)

        for h_f in range(f_h):
            for w_f in range(f_w):
                dX[:, :, h_f : h_f + self.n_h_prime * stride : stride, w_f : w_f + self.n_w_prime * stride : stride] += dX_windows[:, :, :, :, h_f, w_f]

        return dX

    def parameter_gradient(self, X_2D: np.ndarray, dZ: np.ndarray) -> None:
        """
        """

        m = dZ.shape[0]

        dZ = dZ.transpose(1, 0, 2, 3)

        dZ_2D = dZ.reshape(self.n_filters, m * self.n_h_prime * self.n_w_prime)

        dW_2D = dZ_2D @ X_2D

        self.dW = dW_2D.reshape(self.n_filters, self.n_c, self.f_h, self.f_w)

        #self.b = np.sum(dZ, axis=(0, 2, 3), keepdims=True)

    def input_gradient(self, W: np.ndarray, dZ: np.ndarray) -> np.ndarray:
        """
        """

        m = dZ.shape[0]

        dZ = dZ.transpose(0, 2, 3, 1)
        dZ_2D = dZ.reshape(m * self.n_h_prime * self.n_w_prime, self.n_filters)

        W_2D = W.reshape(self.n_filters, self.n_c * self.f_h * self.f_w)

        dX_2D = dZ_2D @ W_2D

        dX = self.col2im(dX_2D, self.n_c, self.f_h, self.f_w, self.stride)

        return dX

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        """

        Z = self.cross_correlation(X, self.W, self.stride, self.padding) #+ self.b

        return Z

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        """

        self.parameter_gradient(self.X_2D, dZ)
        
        dX_padded = self.input_gradient(self.W, dZ)
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
        self.mp_h = None
        self.mp_w = None
        self.stride = stride
        self.max_indices = None

    def max_pooling(self, A: np.ndarray, p_h: int, p_w: int, stride: int) -> np.ndarray:
        """
        """

        self.A = A
        m, n_filters, n_h_prime, n_w_prime = A.shape[0], A.shape[1], A.shape[2], A.shape[3]

        self.mp_h = ((n_h_prime - p_h) // stride) + 1
        self.mp_w = ((n_w_prime - p_w) // stride) + 1

        A_windows = sliding_window_view(A, (p_h, p_w), axis=(2, 3))

        A_windows = A_windows[:, :, ::stride, ::stride, :, :]

        windows = A_windows.reshape(m, n_filters, self.mp_h, self.mp_w, p_h * p_w)

        self.max_indices = np.argmax(windows, axis=4) #no keepdims=True, we want to remove this axis altogether

        A_pooled = np.max(A_windows, axis=(4, 5))

        return A_pooled

    def pooled_gradient(self, dA_pooled: np.ndarray) -> np.ndarray:
        """
        """

        dA = np.zeros_like(self.A)

        max_rows, max_cols = np.unravel_index(self.max_indices, (self.p_h, self.p_w))


        for h_p in range(self.p_h):
            for w_p in range(self.p_w):
                offset_match = (max_rows == h_p) & (max_cols == w_p)
                dA[:, :, h_p : h_p + self.mp_h * self.stride : self.stride, w_p : w_p + self.mp_w * self.stride : self.stride] += (dA_pooled * offset_match)

        return dA

    def forward(self, A: np.ndarray) -> np.ndarray:
        """
        """

        A_pooled = self.max_pooling(A, self.p_h, self.p_w, self.stride)

        return A_pooled

    def backward(self, dA_pooled: np.ndarray) -> np.ndarray:
        """
        """

        dA = self.pooled_gradient(dA_pooled)

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