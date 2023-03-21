if '__file__' in globals(): # 현재 수행중인 코드를 담고 있는 파일이 위치한 path 가 전역 변수들의 집합 내에 있는가
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 현재 파일이 위치한 디렉터리의 부모 디렉터리를 모듈 검색 경로에 추가

from itertools import takewhile
import numpy as np
from math import factorial
# import matplotlib.pyplot as plt

from cores import Variable
from cores import plot_dot_graph
from cores import sin, cos, tanh, transpose, sum, matmul

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

def rosenbrock(x0, x1):
    y = 100*(x1-x0**2)**2 + (1-x0)**2
    return y

def four(x):
    y = x**4 - 2*x**2
    return y

def gx2(x):
    return 12*x**2 - 4

if __name__=='__main__':

    x = Variable(np.random.randn(2,3))
    W = Variable(np.random.randn(3,3))
    y = matmul(x, W)
    y.backward()
    print(x.grad.shape)
    print(W)
    
    # x = Variable(np.array([[1,2,3],[4,5,6]]))
    # y = sum(x, axis=0)
    # y.backward()
    # print(y)
    # print(x.grad)

    # x = Variable(np.random.randn(2,3,4,5))
    # y = x.sum(keepdims=True)
    # print(y.shape)
    # c = Variable(np.array([[10,20,30],[40,50,60]]))
    # t = x + c
    # y = sum(t)
    # y.backward()
    # print(y.grad)
    # print(t.grad)
    # print(c.grad)
    # print(x.grad)
    
    # lr=0.001
    # iters = 6

    # x = Variable(np.array(1.0)) # 구간 내에 숫자 채워주기
    # y = tanh(x)
    # y.backward(create_graph=True)
    
    # logs = [y.data]
    
    # for i in range(iters):
    #     gx = x.grad
    #     x.cleargrad()
    #     gx.backward(create_graph=True)

    # gx = x.grad
    # gx.name = 'gx' + str(iters+1)
    # plot_dot_graph(gx, verbose=False, to_file=r'steps\tanh.png')
    
    # # 그래프 그리기
    
    # labels = ['y=sin(x)', r"y'", r"y\''", r"y'''"]
    # for i, v in enumerate(logs):
    #     plt.plot(x.data, logs[i], label=labels[i])
    # plt.legend(loc='lower right')
    # plt.show()