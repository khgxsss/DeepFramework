import os
import subprocess

import cores.cuda

def _dot_func(f): # Core 함수를 DOT 언어로 변환
    dot_func = '{} [label="{}", color="lightblue", style="filled", shape="box"]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    dot_edge = '{} -> {}\n'
    for x in f.inputs: # 함수와 입력 변수의 관계
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs: # 함수와 출력 변수의 관계
        txt += dot_edge.format(id(f), id(y())) # y는 weakref
    return txt

def _dot_var(v, verbose=False): # Variable 인스턴스를 건네면 그 내용을 DOT 언어로 작성된 문자열로 바꿔서 반환
    dot_var = '{} [label="{}", color="orange", style="filled"]\n'
    
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name) # 변수 노드에 고유한 ID 부여

def get_dot_graph(output, verbose=True): # backward method likely (미분 가밧 대 신 DOT 언어로 기술한 문자열을 txt 에 추가)
    txt = ''
    funcs = []
    seen_set = set() # 중복 감지용

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x:x.generation) # 노드를 추적하는 순서는 문제가 되지 않으므로 generation 값으로 정렬하는 코드 주석 처리
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose)
    
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            
            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)
    
    # dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.path.expanduser('~'), '.deepcore') # 홈 디렉터리(로그인 계정) -> 경로 추가
    if not os.path.exists(tmp_dir): # ~/.core 디렉터리가 없다면 새로 생성
        os.makedirs(tmp_dir) # makedirs 경로 내 모든 폴더 가능 <-> mkdir 폴더 하나
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    
    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    # dot 명령 호출
    extension = os.path.splitext(to_file)[1][1:] # 확장자 출력
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

# Utility functions for numpy

def logsumexp(x, axis=1):
    xp = cores.cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m

def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for cores.functions.sum's backward.
    Args:
        gy (cores.core.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.
    Returns:
        cores.core.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.
    Args:
        x (ndarray): Input array.
        shape:
    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

#