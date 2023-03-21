import numpy as np

from cores.core import Function
from cores.core import as_variable
from cores.utils import sum_to as st, reshape_sum_backward 



# 계산용 추가 정의 클래스

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
        
    def backward(self, gy):
        x, = self.inputs # 순전파 시 저장된 값
        return gy * cos(x) # gy = L, np.cos(x) = sindx

class Square(Function):
    def __init__(self) -> None:
        self
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx # ndarray 인스턴스, 출력 쪽에서 전해지는 미분값을 전달
    
class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y, = self.outputs # tanh(x)의 미분값은 1-y^2
        y = y()
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

def square(x):
    return Square()(x)

def tanh(x):
    return Tanh()(x)
    
#

# Loss

class MeanSquaredError(Function):
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

#

# Loss funcs

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

#

# Affine Transformation

class Linear(Function):
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
    
class Sigmoid(Function):
    def forward(self, x):
        # y = 1 / (1 + exp(-x))
        y = x.tanh(x*0.5)*0.5+0.5 # tanh 쌍곡선으로 대체
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

class BroadcastTo(Function): # sum_to 와 상호의존적
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape # numpy의 shape 사용
        y = x.reshape(self.shape) # numpy의 reshape 사용
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gyr = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gyr, self.x_shape)
        return gx

class SumTo(Function): # broadcast_to 와 상호의존적
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = st(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

class Transpose(Function):
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
        return as_variable(x)
    return BroadcastTo(shape)(x)

def matmul(x, W):
    return MatMul()(x, W)

def reshape(x, shape):
    if x.shape == shape: # 아무 일도 하지 않고 x를 그대로 돌려줌 (Variable instance 보장)
        return as_variable(x)
    return Reshape(shape)(x)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

def transpose(x, axes=None):
    return Transpose(axes)(x)