if '__file__' in globals(): # 현재 수행중인 코드를 담고 있는 파일이 위치한 path 가 전역 변수들의 집합 내에 있는가
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 현재 파일이 위치한 디렉터리의 부모 디렉터리를 모듈 검색 경로에 추가

from itertools import takewhile
from math import factorial
import numpy as np

from core import Variable
from core import plot_dot_graph
from core import sin

def doublea(x):
    return x**3 + x**2 + x

def numerical_diff(f, x, eps=1e-4):
    x0 = x - eps
    x1 = x + eps
    y0 = f(x0)
    y1 = f(x1)
    return (y1 - y0) / (2*eps)

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19-14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
            (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

def my_sin(x, number, threshold=0.0001):
    y = 0
    for i in range(0, number):
        t = (-1)**i * x**(2*i+1) / factorial(2*i + 1)
        y += t
        if abs(t.data) < threshold:
            break
    return y

if __name__=='__main__':
    
    x = Variable(np.array(np.pi/4))
    y = my_sin(x, 10000000, threshold=1e-150)
    y.backward()
    
    x.name = 'x'
    y.name = 'z'
    plot_dot_graph(y, verbose=False, to_file='.\steps\goldstein.png')