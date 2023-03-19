import unittest

from step01 import *

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected) # 주어진 두 객체가 동일한지 여부를 따짐
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(expected, x.grad)
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # 무작위 입력값 생성
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad, rtol=1e-05, atol=1e-08) # 두 값이 가까운지 판정 |a-b| <= (atol + rtol * |b|) => bool 값으로 반환
        self.assertTrue(flg)

unittest.main()

# python -m unittest discover tests => tests 폴더 내부 전부 검사해서 unittest 실행

#============================================================================================

if '__file__' in globals(): # 현재 수행중인 코드를 담고 있는 파일이 위치한 path 가 전역 변수들의 집합 내에 있는가
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 현재 파일이 위치한 디렉터리의 부모 디렉터리를 모듈 검색 경로에 추가

import numpy as np

from cores import Variable

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

if __name__=='__main__':
    
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    z.backward()
    print(x.grad, y.grad)