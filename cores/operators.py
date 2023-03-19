from cores import as_array, Function

class Setup_Variable:
    
    # 연산자 오버로딩
    
    def __mul__(self, other): # multiply 곱셈 오버로드 
        return mul(self, other)

    def __rmul__(self, other): # multiply 곱셈 오버로드 
        return mul(self, other)
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(self, other)
    
    def __neg__(self):
        return neg(self)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)
    
    def __pow__(self, other):
        return pow(self, other)
    
    #

# 연산자 오버로딩 클래스

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y # (y,) == return y, 튜플로 출력
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        return gx0, gx1

class Mul(Function):
    def forward(self, x0, x1):
        return x0*x1
    
    def backward(self, gy):
        x0, x1 = self.inputs # simple 버전에서는 ndarray instance 꺼내서 썼으나, 이제는 인스턴스 바로사용
        return x1*gy, x0*gy

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        return gx0, gx1
    
class Div(Function):
    def forward(self, x0, x1):
        return x0/x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

#

# 연산자 오버로딩 함수

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x0, x1 = as_array(x0), as_array(x1)
    return Sub()(x0, x1)

def div(x0, x1):
    x0, x1 = as_array(x0), as_array(x1)
    return Div()(x0, x1)

def pow(x, c):
    return Pow(c)(x)

#