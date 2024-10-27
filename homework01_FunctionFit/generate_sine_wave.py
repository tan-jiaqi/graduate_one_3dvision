import numpy as np
import torch

np.random.seed(2)

T = 20 # 周期
L = 1000 # 信号长度
N = 100 # 信号数量

x = np.empty((N, L), 'int64')
# 生成N个信号,构建数据集
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('datasets/traindata.pt', 'wb'))

