# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# x, y 값
X = np.arange(-1.0, 1.0, 0.2)   # 원소는 10개
Y = np.arange(-1.0, 1.0, 0.2)

# 출력을 저장하는 10X10 그리드
Z = np.zeros((10,10))

# 가중치 (weight)
w_im = np.array([[4.0, 4.0],
                 [4.0, 4.0]])   # 은닉층 2X2 행렬
w_mo = np.array([[1.0],
                [-1.0]])        # 출력층 2X1 행렬

# 편향 (bias)
b_im = np.array([3.0, -3.0])    # 은닉층
b_mo = np.array([0.1])          # 출력층

# 은닉층 (hidden(middle) layer)
def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))     # 시그모이드 함수

# 출력층 (output layer)
def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return u                    # 항등함수

# 그리드맵의 각 그리드별 신경망 연산
for i in range(10):
    for j in range(10):
        
        # 순전파
        inp = np.array([X[i], Y[j]])            # 입력층
        mid = middle_layer(inp, w_im, b_im)     # 은닉층
        out = output_layer(mid, w_mo, b_mo)     # 출력층
        
        # 그리드맵에 신경망 출력값 저장
        Z[j][i] = out[0]
        
# 그리드맵으로 표시
plt.imshow(Z, "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()
plt.show()
