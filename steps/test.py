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