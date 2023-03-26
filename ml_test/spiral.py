if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib.pyplot as plt

import cores
from cores import optimizers
import cores.functions as F
from cores.models import MLP


# 하이퍼파라미터 설정
max_epoch = 300 # 총 준비된 데이터셋 * 300번 사용
batch_size = 30 # 미니배치(모집단에서 표본) 30 * 10
hidden_size = 10 # 은닉층 수
lr = 1.0

# 데이터 읽기 / 모델, 옵티마지어 생성
x, t = cores.datasets.get_spiral(train=True) # train=True -> 학습데이터 리턴, False -> 테스트용 데이터 리턴 | (300,2)입력데이터 (300,)정답레이블
model = MLP((hidden_size, 3)) # 은닉층, 출력층
optimizer = optimizers.SGD(lr).setup(model)
data_size = len(x)
max_iter = math.ceil(data_size/batch_size) # 반복횟수

for epoch in range(max_epoch): # max_epoch 의 각 표본
    # 데이터셋의 인덱스 뒤섞기
    index = np.random.permutation(data_size) # data_size -1까지의 int가 rand하게 배열된 list
    sum_loss = 0

    for i in range(max_iter):
        
        # 미니배치 생성
        batch_index = index[i*batch_size:(i+1)*batch_size] # 방금 생성한 index에서 앞에서부터 차례로 꺼내 사용함
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t) # 손실함수 설정
        model.cleargrads() # 매개변수 초기화
        loss.backward() # 역전파법
        optimizer.update() # 경사하강법 적용
        sum_loss += float(loss.data) * len(batch_t)
    
    # epoch마다 학습 경과 출력
    avg_loss = sum_loss / data_size
    print(f'epoch {epoch+1}, loss {avg_loss}')

# Plot boundary area the model predict
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with cores.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()