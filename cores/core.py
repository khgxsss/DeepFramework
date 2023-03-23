import contextlib
import numpy as np
import weakref

import cores.functions

class Setup_Variable1:
    # Variable 연산자 오버로딩

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

class Setup_Variable2(Setup_Variable1):
    def __len__(self): # 인스턴스에 대해서도 len 함수를 사용할 수 있게
        return len(self.data)
    
    def __getitem__(self, indices):
        return cores.functions.get_item(self, indices)
    
    def __repr__(self) -> str: # variable print 했을 때 나올 수 있게
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' +p+')'

class Setup_Variable3(Setup_Variable2):
    @property # shape 메서드를 인스턴스 변수처럼 사용할 수 있음 : x.shape() 대신 x.shape로 호출
    def shape(self): # 모양
        return self.data.shape
    
    @property
    def ndim(self): # 차원 수
        return self.data.ndim
    
    @property
    def size(self): # 원소 수
        return self.data.size
    
    @property
    def dtype(self): # 데이터 타입(미지정 시 flaot64 또는 int64로 초기화)
        return self.data.dtype
    
    @property
    def T(self): # 전치행렬
        return cores.functions.transpose(self)
    
class Setup_Variable4(Setup_Variable3):
    # Tensor
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return cores.functions.reshape(self, shape)
    
    def sum(self, axis=None, keepdims=False):
        return cores.functions.sum(self, axis, keepdims)
    
    def transpose(self, *axes): # 축 순서 지정 가능하게
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return cores.functions.transpose(self, axes)
    
    #
    
class Variable(Setup_Variable4):
    __array_prioty__ = 200

    def __init__(self, data, name=None): # data와 grad는 모두 넘파이 다차원 배열
        if data is not None:
            if not isinstance(data, np.ndarray): # np.ndarray인지 검출
                raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')
            
        self.data = data
        self.name = name # name으로 변수들 구분. 계산 그래프 시각화 등에 사용
        self.grad = None # 미분값은 실제 역전파시 미분값을 계산하여 대입
        self.creator = None # 변수를 만들어준 함수 저장
        self.generation = 0 # 세대 저장(부모노드와 자식노드간 계산 우선순위 설정)
    
    def set_creator(self, func):
        self.creator = func # creator 함수 저장
        self.generation = func.generation+1 # 세대를 기록한다(부모 세대 + 1)
    
    def cleargrad(self): # 미분값 초기화 메서드 : 같은 변수를 사용해 다른 계산을 할 경우 초기화 필요
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False): # retain_grad = 중간 변수 미분값 모두 None (계산 과정에서 각 함수의 출력 변수의 미분값 유지하지 않음) x.backward() 호출하면 x만 미분값 유지
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data)) # self.data와 형상과 데이터 타입이 같은 ndarray instance 생성 : 모든 요소를 1로 채워서 돌려줌 ( 스칼라이면 스칼라 )
        
        funcs = []
        seen_set = set()

        def add_func(f): # 함수 리스트를 세대 순으로 정렬하는 역할 => pop은 자동으로 세대가 가장 큰 함수 꺼냄
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                try:
                    funcs.sort(key=lambda x: x.generation) # x.generation을 key로 사용해 정렬
                except AttributeError as e:
                    print(str(e) + '\nmaybe you can turn on create_graph option')
                    exit(0)
        
        add_func(self.creator)

        while funcs: # 역전파 시 creator 함수가 None일 때 까지 자동 미분
            f = funcs.pop() # 함수를 가져온다. : 리스트의 맨 뒤 원소
            gys = [output().grad for output in f.outputs] # 출력 변수인 outputs에 담겨 있는 미분값들을 리스트에 담음. output()는 weakref로 인해 참조하는 함수로 변경되었음

            with using_config('enable_backprop', create_graph): # 역전파 비활성 모드(단 1회만 실행), 2차 이상의 미분 필요하면 True
                gxs = f.backward(*gys) # 메인 backward
                if not isinstance(gxs, tuple): # gxs가 튜플이 아니라면 튜플로 변환
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs): # 역전파로 전파되는 미분값을 Variable의 인스턴스 변수 grad에 저장. i번째 원소에 대해 f.inputs[i]의 미분값은 gxs[i]에 대응 => 튜플로 출력
                    if x.grad is None: # 미분값을 처음 설정하는 경우에는 출력 쪽 미분값을 그대로 대입
                        x.grad = gx
                    else: # 다음번부터는 전달된 미분값을 더해줌
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator) # 한 순서 앞의 함수를 리스트에 추가
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y는 약한 참조 => 참조 카운트가 0이 되어 미분값 데이터가 메모리에서 삭제됨

class Parameter(Variable): # Variable과 기능은 같지만 Parameter 인스턴스(매개변수)만을 담는 구조
    pass

class Function: # Define-by-Run 구조 구현 : Linked List
    def __call__(self, *inputs) -> any: # 파라미터를 모아서 받음 (*)
        inputs = [as_variable(x) for x in inputs] # 모두 variable 인스턴스로 변환

        xs = [x.data for x in inputs] # inputs 리스트의 각 원소 x에 대해 x.data를 꺼내고 꺼낸 원소들로 새로운 리스트 작성
        ys = self.forward(*xs) # 구체적 계산 | xs = [x0, x1] 일때 self.forward(*xs)를 하면 self.forward(x0, x1)과 동일한 동작
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # output이 scalar 여도 numpy.float 형식이 아닌 ndarray로 나오도록 변환
        
        if Config.enable_backprop: # 역전파 기능 on
            self.generation = max([x.generation for x in inputs]) # 입력 변수가 둘 이상이라면 functions의 generation은 max 선택 : 기본적으론 입력 variable의 generation 따라감
            for output in outputs:
                output.set_creator(self) # 출력 변수에 creator 함수 저장
            
            self.inputs = inputs # 입력 변수(Variable) 유지(순전파 때의 결괏값)
            self.outputs = [weakref.ref(output) for output in outputs ] # 출력 자체도 저장, 약한 참조를 사용해 가비지 콜렉터가 약한 참조가 남아있어도 output을 파괴하고 메모리를 재할당
        
        return outputs if len(outputs) > 1 else outputs[0] # 리스트의 원소가 하나라면 첫 번째 원소를 반환한다.
    
    def forward(self, xs):
        raise NotImplementedError() # 이 method는 상속하여 구현해야 한다.
    
    def backward(self, gys):
        raise NotImplementedError()

# 연산자 오버로딩 클래스

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y # (y,) == return y, 튜플로 출력
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = cores.functions.sum_to(gx0, self.x0_shape)
            gx1 = cores.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, x0, x1):
        return x0*x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = cores.functions.sum_to(gx0, x0.shape)
            gx1 = cores.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = cores.functions.sum_to(gx0, self.x0_shape)
            gx1 = cores.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Div(Function):
    def forward(self, x0, x1):
        return x0/x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = cores.functions.sum_to(gx0, x0.shape)
            gx1 = cores.functions.sum_to(gx1, x1.shape)
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
    x0, x1 = as_array(x0), as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x0, x1 = as_array(x0), as_array(x1)
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

class Config: # 설정 값은 클래스 상태로 이용 -> 단 하나만
    enable_backprop = True # 역전파 활성 모드 on?

# config용 함수

@contextlib.contextmanager # 데코레이터로 try ~ finally 전후로 문맥:context를 판단해서 yield 전 전처리 로직, yield 후 후처리 로직 작성
def using_config(name, value): # 사용할 Config 속성의 이름 name을 가리킴(str), with 블록에 들어감
    old_value = getattr(Config, name) # name을 getattr 함수에 넘겨 Config 클래스에서 꺼내옴
    setattr(Config, name, value) # 새로운 값 설정
    try:
        yield
    finally:
        setattr(Config, name, old_value) # with 블록을 빠져나오면서 원래 값으로 복원됨
    
def no_grad(): # using_config False 값 넣을 때의 편의성 함수
    return using_config('enable_backprop', False)

#

# 데이터 구조 변화용 함수

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj): # 다른 형식 인스턴스도 Variable로 통일해서 연산이 사용 가능하게
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

#