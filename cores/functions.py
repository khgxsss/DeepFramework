import numpy as np

from cores import Function
from cores.core import as_variable



# 계산용 추가 정의 클래스

class Square(Function):
    def __init__(self) -> None:
        self
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx # ndarray 인스턴스, 출력 쪽에서 전해지는 미분값을 전달
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x, = self.inputs
        gx = np.exp(x) * gy
        return gx

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
        
    def backward(self, gy):
        x, = self.inputs # 순전파 시 저장된 값
        return gy * cos(x) # gy = L, np.cos(x) = sindx

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y, = self.outputs # tanh(x)의 미분값은 1-y^2
        y = y()
        return gy * (1 - y * y)

#

# 계산용 추가 정의 함수

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

#

# Tensor

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape # numpy의 shape 사용
        y = x.reshape(self.shape) # numpy의 reshape 사용
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)

# class Sum(Function):
#     def forward(self, x):
#         y = x.sum()
#         return 
    
#     def backward(self, gy):
#         gx = broadcast_to(gy, self.x_shape)
#         return gy
#

#

# Tensor funcs

def reshape(x, shape):
    if x.shape == shape: # 아무 일도 하지 않고 x를 그대로 돌려줌 (Variable instance 보장)
        return as_variable(x)
    return Reshape(shape)(x)