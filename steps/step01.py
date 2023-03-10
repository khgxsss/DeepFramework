import numpy as np

class Variable:
    def __init__(self, data) -> None: # data와 grad는 모두 넘파이 다차원 배열
        if data is not None:
            if not isinstance(data, np.ndarray): # np.ndarray인지 검출
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.grad = None # 미분값은 실제 역전파시 미분값을 계산하여 대입
        self.creator = None # 변수를 만들어준 함수 저장
    
    def set_creator(self, func):
        self.creator = func # creator 함수 저장
    
    ## 재귀 버전
    # def backward(self):
    #     f = self.creator # 함수를 가져온다.
    #     if f is not None: # 역전파 시 creator 함수가 None일 때 까지
    #         x = f.input # 함수의 입력을 가져온다.
    #         x.grad = f.backward(self.grad) # 함수의 backward method를 호출한다.
    #         x.backward() # 하나 앞 변수의 backward method를 호출한다(재귀)
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # self.data와 형상과 데이터 타입이 같은 ndarray instance 생성 : 모든 요소를 1로 채워서 돌려줌 ( 스칼라이면 스칼라 )
        funcs = [self.creator]
        while funcs: # 역전파 시 creator 함수가 None일 때 까지
            f = funcs.pop() # 함수를 가져온다. : 리스트의 맨 뒤 원소
            x, y = f.input, f.output # 함수의 입력과 출력 가져온다.
            x.grad = f.backward(y.grad) # 함수의 backward method를 호출한다.
            
            if x.creator is not None:
                funcs.append(x.creator) # 한 순서 앞의 함수를 리스트에 추가

class Function: # Define-by-Run 구조 구현 : Linked List
    def __call__(self, input: any) -> any:
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) # output이 scalar 여도 numpy.float 형식이 아닌 ndarray로 나오도록 변환
        output.set_creator(self) # 출력 변수에 creator 함수 저장
        self.input = input # 입력 변수(Variable) 유지
        self.output = output # 출력 자체도 저장
        return output
    
    def forward(self, x):
        raise NotImplementedError() # 이 method는 상속하여 구현해야 한다.
    
    def backward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx # ndarray 인스턴스, 출력 쪽에서 전해지는 미분값을 전달
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

if __name__ == "__main__":
    # testcode
    x = Variable(np.array(1))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)