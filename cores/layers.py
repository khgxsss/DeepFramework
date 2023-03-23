import numpy as np
import weakref

from cores import Parameter
from cores.functions import linear

class Layer: # Function과 마찬가지로 변수를 변환하지만 매개변수를 유지하고 이를 사용하여 변환을 하는 클래스
    def __init__(self):
        self._params = set()
    
    def __call__(self, *inputs): # 입력받은 인수를 건네 forward 메서드 호출
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple): # 출력이 하나뿐이라면 튜플이 아닌 직접 반환
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outpus = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def __setattr__(self, name, value) -> None: # instance 변수를 설정할 때 호출되는 메서드. name이 이름인 인스턴스 변수에 값으로 value 전달
        if isinstance(value, (Parameter, Layer)): # 액자구조 Layer - Layer 구현
            self._params.add(name) # Parameter와 Layer 인스턴스의 이름이 추가
        super().__setattr__(name, value)
    
    def forward(self, inputs): # 자식 클래스에서 구현
        raise NotImplementedError
    
    def params(self): # Layer 인스턴스에 담겨있는 Parameter 인스턴스들을 꺼내줌
        for name in self._params:
            obj = self.__dict__[name] # 모든 인스턴스 변수가 dict 타입으로 저장:Parameter instance만 꺼내 쓸 수 있음
            
            if isinstance(obj, Layer): # obj가 Layer 인스턴스라면 obj.param() 호출
                yield from obj.params() # Layer 속 Layer에서도 매개변수를 재귀적 호출 (yield from : generator을 사용해 또 다른 generator 만듦)
            else:
                yield  obj
    
    def cleargrads(self): # 모든 매개변수의 기울기를 재설정
        for param in self.params():
            param.cleargrad()

# 선형 변환

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size # 입력 크기
        self.out_size = out_size # 출력 크기
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None: # 지정되어 있지 않으면 forward로 연기
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
        
    def _init_W(self): # 가중치 W 생성
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    
    def forward(self, x):
        # 데이터를 흘려보내는 시점에 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        
        y = linear(x, self.W, self.b)
        return y