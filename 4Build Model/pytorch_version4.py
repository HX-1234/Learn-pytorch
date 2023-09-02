"""
作者：黄欣
日期：2023年09月01日
"""
from collections import OrderedDict

import torch
from torch import nn

#############################################
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

print(model)
#############################################

#############################################
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
print(model)
conv1 = model._modules['conv1']
print(conv1)
#############################################


#############################################
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for layer in self.linears:
            x = layer(x)
        return x

model = MyModule()
print(model)
#############################################


#############################################
class MyModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

model = MyModule2()
X = torch.rand(10,20,20) # CHW
y = model(X,'conv','lrelu')
print(model)
print(y.shape)
#############################################

#############################################
m = nn.Conv2d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape)
#############################################

#############################################
m = nn.MaxPool2d(3, stride=2)
input = torch.randn(20, 16, 50, 32)
output = m(input)
print(output.shape)
#############################################

#############################################
m = nn.ReLU()
input = torch.tensor(range(-5,5))
output = m(input)
print(output)
#############################################

#############################################
m = nn.Softmax(dim=1)
input = torch.randn(3, 2)
output = m(input)
print(output)
#############################################
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)
print(output.shape)
#############################################

#############################################
m = nn.Dropout(p=0.9)
input = torch.randn(3, 5)
output = m(input)
print(output)
#############################################

#############################################
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()



