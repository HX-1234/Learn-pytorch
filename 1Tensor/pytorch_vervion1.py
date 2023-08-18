"""
作者：黄欣
日期：2023年08月18日
"""

import torch
import numpy as np

# data = [[1, 2],[3, 4]]
# print(type(data[0][0]))
# x_data = torch.tensor(data)
# print(x_data.dtype)
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
# print(x_np)

# t = torch.ones(5)
# print(f"t: {t}")
# n = t.numpy()
# print(f"n: {n}")
#
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")

# n = np.ones(5)
# t = torch.from_numpy(n)
# print(f"t: {t}")
# print(f"n: {n}")
# np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")

# data = [[1, 2],[3, 4]]
# x_data = torch.tensor(data)
# x_ones = torch.ones_like(x_data)
# print(f"\n {x_ones} \n")
# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"\n {x_rand} \n")

# rand_tensor = torch.rand(3,2)
# ones_tensor = torch.ones(3,2)
# print(rand_tensor)
# print(ones_tensor)

# tensor = torch.rand(3, 4)
# print(f"Shape: {tensor.shape}")
# print(f"Datatype: {tensor.dtype}")
# print(f"Device: {tensor.device}")
# print(f"转置: {tensor.T}")
#
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')
#   print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(3,3)
tensor[:,1] = 0
print(tensor)
tensor.add_(5)
print(tensor)

# 按元素乘法
# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)}")
# print(f"tensor * tensor \n {tensor * tensor}")
# # 矩阵乘法
# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)}")
# print(f"tensor @ tensor.T \n {tensor @ tensor.T}")




