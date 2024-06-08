import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

input = np.random.rand(4, 2) # 入力データ
correct = np.random.rand(4, 1) # 正解データ
print(input)
print(correct)
a = [[1] for _ in range(4)]
train_lebel = torch.FloatTensor(a) # pytorchで扱える配列に変更

print(train_lebel)

input = torch.FloatTensor(input) # pytorchで扱える配列に変更
correct = torch.FloatTensor(correct) # pytorchで扱える配列に変更
print(input, correct)