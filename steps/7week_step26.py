if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

# 변수 생성
x = Variable(np.array(1.0))
x.name = 'x'

y = Variable(np.array(1.0))
y.name = 'y'

# 계산 수행
z = goldstein(x, y)
z.name = 'z'

# 역전파 수행
z.backward()

# 계산 그래프 출력
plot_dot_graph(z, verbose=True, to_file='goldstein.png')
