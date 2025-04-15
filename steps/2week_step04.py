import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Addone(Function):
    def forward(self, x):
        return x + 1


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# 기존 함수들의 numerical differentiation 테스트
f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print("Square 함수의 x=2에서의 미분값:", dy)


def f1(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f1, x)
print("f(x) = (exp(x^2))^2 at x=0.5의 미분값:", dy)


# 새로운 함수 y = (1 + e^x)^2 추가
class CustomFunction(Function):
    def forward(self, x):
        return (1 + np.exp(x)) ** 2


x_value = Variable(np.array(1.0))
f2 = CustomFunction()
dy_custom = numerical_diff(f2, x_value)
print("y = (1 + e^x)^2 at x=1의 미분값:", dy_custom)
