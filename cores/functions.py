import numpy as np

from cores import Function

# 계산용 추가 정의 클래스

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx # ndarray 인스턴스, 출력 쪽에서 전해지는 미분값을 전달
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x = self.inputs[0].data # 순전파 시 저장된 값
        return gy * np.cos(x) # gy = L, np.cos(x) = sindx

#

# 계산용 추가 정의 함수

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def sin(x):
    return Sin()(x)

#