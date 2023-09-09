"""
作者：黄欣
日期：2023年09月09日
"""

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

###################################################################
# z = torch.matmul(x, w)+b
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#
# loss.backward()
# print(w.grad)
# print(b.grad)
###################################################################

##################################################################
# z = torch.matmul(x, w)+b
# print(z.requires_grad)
#
# with torch.no_grad():
#     z = torch.matmul(x, w)+b
# print(z.requires_grad)
# print(b.requires_grad)
# print(w.requires_grad)
##################################################################

##################################################################
# z = torch.matmul(x, w)+b
# z_det = z.detach()
# print(z_det.requires_grad)
##################################################################

a = torch.tensor([1.,2.,3.,4],requires_grad=True)
b = a ** 2
c = b.mean()
d = b.sum()

c.backward(retain_graph=True)#
print(a.grad)
d.backward()
print(a.grad)







