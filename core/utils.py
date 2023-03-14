import numpy as np
from core_simple import Variable

def _dot_var(v, verbose=False): # Variable 인스턴스를 건네면 그 내용을 DOT 언어로 작성된 문자열로 바꿔서 반환
    dot_var = '{} [label={}, color=orange, style=filled]\n'
    
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name) # 변수 노드에 고유한 ID 부여

def _dot_func(f): # Core 함수를 DOT 언어로 변환
    dot_func = '{} [label={}, color=lightblue, style=filled, shape=box\n]'
    txt = dot_func.format(id(f), f.__class__.__name__)
    
    dot_edge = '{} -> {}\n'
    for x in f.inputs: # 함수와 입력 변수의 관계
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs: # 함수와 출력 변수의 관계
        txt += dot_edge.format(id(f), id(y())) # y는 weakref
    return txt

if __name__=='__main__':
    x = Variable(np.random.randn(2,3))
    y = Variable(np.random.randn(2,3))
    z = x + y
    txt = _dot_func(y.creator)
    print(_dot_var(txt))

