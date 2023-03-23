import numpy as np

import cores.core
import cores.cuda
import cores.utils



# 계산용 추가 정의 클래스

class Cos(cores.core.Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Exp(cores.core.Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

class Log(cores.core.Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

class Sin(cores.core.Function):
    def forward(self, x):
        return np.sin(x)
        
    def backward(self, gy):
        x, = self.inputs # 순전파 시 저장된 값
        return gy * cos(x) # gy = L, np.cos(x) = sindx
    
class Tanh(cores.core.Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y = self.outputs[0]() # tanh(x)의 미분값은 1-y^2, weakref
        return gy * (1 - y * y)

#

# 계산용 추가 정의 함수

def cos(x):
    return Cos()(x)

def exp(x):
    return Exp()(x)

def log(x):
    return Log()(x)

def sin(x):
    return Sin()(x)

def tanh(x):
    return Tanh()(x)
    
#

# Loss

class LogSoftmax(cores.core.Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        log_z = cores.utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx

class MeanSquaredError(cores.core.Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. /len(diff))
        gx1 = -gx0
        return gx0, gx1

class Softmax(cores.core.Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cores.cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx
    
class SoftmaxCrossEntropy(cores.core.Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = cores.utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cores.cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

#

# Loss funcs

def log_softmax(x, axis=1):
    return LogSoftmax(axis)(x)

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def softmax(x, axis=1):
    return Softmax(axis)(x)

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

#

# Affine Transformation

class Linear(cores.core.Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb
    
class Sigmoid(cores.core.Function):
    def forward(self, x):
        # y = 1 / (1 + np.exp(-x))
        y = np.tanh(x*0.5)*0.5+0.5 # tanh 쌍곡선으로 대체
        return y
    
    def backward(self, gy):
        y, = self.outputs
        y = y()
        return gy*y*(1-y)

#

#Affine funcs

def linear(x, W, b=None):
    return Linear()(x, W, b)

def sigmoid(x):
    return Sigmoid()(x)

#

# Tensor

class BroadcastTo(cores.core.Function): # sum_to와 상호의존적
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cores.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
class MatMul(cores.core.Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

class Reshape(cores.core.Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape # numpy의 shape 사용
        y = x.reshape(self.shape) # numpy의 reshape 사용
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
class Sum(cores.core.Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gyr = cores.utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gyr, self.x_shape)
        return gx

class SumTo(cores.core.Function): # broadcast_to 와 상호의존적
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = cores.utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

class Transpose(cores.core.Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

#

# Tensor funcs

def broadcast_to(x, shape):
    if x.shape == shape:
        return cores.core.as_variable(x)
    return BroadcastTo(shape)(x)

def matmul(x, W):
    return MatMul()(x, W)

def reshape(x, shape):
    if x.shape == shape: # 아무 일도 하지 않고 x를 그대로 돌려줌 (Variable instance 보장)
        return cores.core.as_variable(x)
    return Reshape(shape)(x)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return cores.core.as_variable(x)
    return SumTo(shape)(x)

def transpose(x, axes=None):
    return Transpose(axes)(x)

# slice

class GetItem(cores.core.Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

class GetItemGrad(cores.core.Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cores.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)

#