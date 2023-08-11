#!/usr/bin/env python
# coding: utf-8

# # Neural Network (신경망)

# ## Regression (회귀)

# In[3]:


# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# x, y 값
X = np.arange(-1.0, 1.0, 0.2)       # 원소는 10개
Y = np.arange(-1.0, 1.0, 0.2)

# 출력을 저장하는 10X10 그리드
Z = np.zeros((10,10))

# 가중치 (weight)
w_im = np.array([[4.0, 4.0],
                 [4.0, 4.0]])       # 은닉층 2X2 행렬
w_mo = np.array([[1.0],
                [-1.0]])            # 출력층 2X1 행렬

# 편향 (bias)
b_im = np.array([3.0, -3.0])        # 은닉층
b_mo = np.array([0.1])              # 출력층

# 은닉층 (hidden(middle) layer)
def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))         # 시그모이드 함수

# 출력층 (output layer)
def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return u                        # 항등함수

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


# ## Classification (분류)

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# x, y 값
X = np.arange(-1.0, 1.0, 0.1)       # 원소의 수는 20개
Y = np.arange(-1.0, 1.0, 0.1)

# 가중치
w_im = np.array([[1.0,2.0],
                 [2.0,3.0]])        # 은닉층 2X2 행렬
w_mo = np.array([[-1.0,1.0],
                 [1.0,-1.0]])       # 출력층 2X2 행렬

# 편향
b_im = np.array([0.3,-0.3])         # 은닉층
b_mo = np.array([0.4,0.1])          # 출력층

# 은닉층
def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))         # 시그모이드 함수

# 출력층
def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return np.exp(u)/np.sum(np.exp(u))      # 소프트맥스 함수

# 분류 결과를 저장하는 리스트
x_1 = []
y_1 = []
x_2 = []
y_2 = []

# 그리드맵의 각 그리드별 신경망 연산
for i in range(20):
    for j in range(20):

        # 순전파
        inp = np.array([X[i], Y[j]])
        mid = middle_layer(inp, w_im, b_im)
        out = output_layer(mid, w_mo, b_mo)

        # 확률의 크기를 비교해 분류함
        if out[0] > out[1]:
            x_1.append(X[i])
            y_1.append(Y[j])
        else:
            x_2.append(X[i])
            y_2.append(Y[j])

# 산포도 표시
plt.scatter(x_1, y_1, marker="+")
plt.scatter(x_2, y_2, marker="o")
plt.show()


# In[ ]:




