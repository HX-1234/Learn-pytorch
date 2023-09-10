# Save, Load and Use Model

## 一、内容

本部分将涉及如何保存和加载模型，包括三种方法：1、只保存和加载参数；2、保存和加载整个模型；3、在训练过程中保存和加载。

## 二、代码

### 1、只保存和加载参数

这样只需保存必要参数，使得模型大小减小，利于保存。

**保存**

```py
torch.save(model.state_dict(), 'model_weights.pth')
```

这里保存上一部分内容中训练的模型。

**加载**

```py
model = NeuralNetwork()
model.load_state_dict(torch.load('model_weights.pth'))
print(model)
```

![image-20230910102024710](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309101020782.png)

### 2、保存和加载整个模型

**保存**

```py
torch.save(model, 'model.pth')
```

**加载完整模型**

```py
model = torch.load('model.pth')
print(model)
```

> 注意：这里不需要model = NeuralNetwork()构建模型结构，因为保存时已经存了模型的结构。

### **3、训练中保存**

**保存**

```py
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    loss = test_loop(test_dataloader, model, loss_fn)
    torch.save({
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join("model_save",str(t)))
print("Done!")
```

![image-20230910105806992](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309101058031.png)

> 每一个epoch都会保存一次模型，并保存代化器。

```py
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

checkpoint = torch.load(os.path.join("model_save",str(0)))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss_last = checkpoint['loss']

# 再接着从0号模型进行训练
model.train()
train_loop(train_dataloader, model, loss_fn, optimizer)
loss_now = test_loop(test_dataloader, model, loss_fn)
print("loss_last", loss_last)
print("loss_now", loss_now)
```

![image-20230910110650006](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309101106046.png)

> 接着从0号模型进行训练，这一次loss比上0号模型loss要小，说明再训练的模型性能是提高了的。