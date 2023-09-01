# Transform

##  一、内容

torchvision.transforms包含多种数据预处理方法：数据中心化、数据标准化、缩放、裁剪、旋转、翻转、填充、噪声添加、灰度变换、线性变换、仿射变换、亮度、饱和度及对比度变换等。

![image-20230831220237264](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202308312202395.png)

![image-20230901074213042](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309010742087.png)

> 上图是transforms处理数据时所处的位置。主要在Dataset中对图像进行预处理。

![image-20230901073922394](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309010739474.png)

> transforms主要完成以上四个部骤。

## 二、代码

### **1、Compose**

transforms.Compose是PyTorch中torchvision.transforms模块的一个类，它的主要作用是将多个图像变换操作组合在一起。这个类的构造很简单，它接受一个由多个Transform对象组成的列表作为参数，然后按照列表中的顺序依次对图像进行变换。

```py
import numpy as np
from PIL import Image
from torchvision import transforms


path = "fashion_mnist_images/test/3/0000.png"
img = Image.open(path)
img = img.convert('RGB')


trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)

print(str(trans))
```

![image-20230901091153522](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309010911557.png)

> 上图是输出的Compose结构。这些参数mean=[0.485, 0.456, 0.406]和std=[0.229, 0.224, 0.225]是ImageNet数据集的均值和标准差1。使用ImageNet的均值和标准差是一种常见的做法，因为它们是根据数百万张图像计算得出的。如果要在自己的数据集上从头开始训练，则可以计算新的均值和标准差。否则，建议使用ImageNet预训练模型及其平均值和标准差。

### **2、transforms_inverse**

这个函数是自己实现的，主要是为了将tensor数据反向转化成PIL格式数据，方便打印，查看各种变换的效果。

```py
path = "dog.jpg"
img = Image.open(path)
plt.imshow(img)
plt.show()

trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)


def transforms_inverse(img, trans):
    # 反向transforms
    if "Normalize" in str(trans): # 如果使用了Normalize
        # trans.transforms返回Compose中的内容
        # trans返回Compose对象
        # filter()它接受一个函数和一个序列作为参数，然后返回一个迭代器，其中包含序列中所有使函数返回值为True的元素。
        # list()函数将返回的迭代器转换为列表
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), trans.transforms))
        # 获得归一化时使用的均值和标准差
        mean = torch.tensor(norm_transform[0].mean)
        std = torch.tensor(norm_transform[0].std)
        # mul_和add_是原位操作
        # std[:, None, None]是一种用于增加张量维度的技巧
        # 形状为(3,)的一维张量转换成形状为(3, 1, 1)的三维张量，其中的3是通道数
        img.mul_(std[:, None, None]).add_(mean[:, None, None])

    img = img.transpose(0,2).transpose(0,1) # CHW => HWC
    # tensor => np
    img = np.array(img) * 255

    # 使用Image.fromarray()函数将数组转换为PIL图像
    if img.shape[2] == 3:# channel==3
        img = Image.fromarray( img.astype('uint8') ).convert('RGB')
    elif img.shape[2] == 1:  # channel==1
        # 当channel==1，必须用squeeze()函数移除多余的维度
        img = Image.fromarray(img.astype('uint8').squeeze())

    return img


img_tensor = trans(img)
img_PIL = transforms_inverse(img_tensor,trans)
plt.imshow(img_PIL)
plt.show()
```

![image-20230901102803610](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011028708.png)

![image-20230901102818963](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011028062.png)

> 前一张是原图，后一张是反向转换后的图，与原图没有区别。

### **3、Resize**

```python
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

> 必须是元组(224,224)

![image-20230901103529127](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011035262.png)

### **4、CenterCrop**

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.CenterCrop(150),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901103822378](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011038425.png)

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.CenterCrop(300),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901103901105](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011039196.png)

> 当transforms.CenterCrop(300)中心的大小大于原图则会在周围补0



### **5、RandomCrop**

![image-20230901104713467](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011047502.png)

**从图片中随机裁剪出尺寸为size的图片**

- **size:**所需裁剪图片尺寸
- **padding :**设置填充大小
  - 当为a时，上下左右均填充a个像素
  - 当为(a, b)时,上下填充b个像素，左右填充a个像素 
  - 当为(a, b, c, d)时,左，上，右,下分别填充a, b, c, d
- **pad_if_need:**若图像小于设定size，则padding
- **padding_mode :**填充模式，有4种模式
  - **constant :**像素值由fill设定
  - **edge :**像素值由图像边缘像素决定
  - **reflect :**镜像填充,最后一个像素不镜像
  - **symmetric :**镜像填充,最后一个像素镜像
- **fill :** constant时,设置填充的像素值

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.RandomCrop(224,padding=26,fill=(255,0,0)),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901105400154](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011054281.png)

```py
transforms.RandomCrop(512,pad_if_needed=True)
```

![image-20230901105550023](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011055080.png)

```py
transforms.RandomCrop(512,pad_if_needed=True,padding_mode='edge')
```

![image-20230901105800402](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011058456.png)

```py
transforms.RandomCrop(512,pad_if_needed=True,padding_mode='reflect')
```

![image-20230901105909502](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011059635.png)

### **6、RandomResizedCrop**

![image-20230901110815228](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011108265.png)

**随机大小、长宽比裁剪图片**

- **size ：**所需裁剪图片尺寸
- **scale：**随机裁剪面积比例，默认区间(0.08, 1)
- **ratio ：**随机长宽比，默认区间(3/4, 4/3)
- **interpolation ：**插值方法
  - **PIL.Image.NEAREST**
  - **PIL.Image.BILINEAR**
  - **PIL.Image.BICUBIC**

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.RandomResizedCrop(224,scale=(0.2,0.2)),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901110658686](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011106803.png)

>  随机裁剪20%的面积

### 7、FiveCrop & TenCrop

![image-20230901110921177](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011109212.png)

**在图像的上下左右以及中心裁剪出尺 寸为size的5张图片，TenCrop对这5张图片 进行水平或者垂直镜像获得10张图片**

* **size :**所需裁剪图片尺寸
* **vertical_flip :**是否垂直翻转

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.FiveCrop(112), # 将获得5张图，是一个tuple元组，要拆开
     transforms.Lambda( lambda crops: torch.stack( [ (transforms.ToTensor()(each)) for each in crops ] ) ),

     # transforms.ToTensor(),
     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
img_tensor = trans(img)

Ncrops, C, H, M = img_tensor.shape
for i in range(Ncrops):
    # transforms.FiveCrop(112),  # 将获得5张图,遍历每张图
    img_PIL = transforms_inverse(img_tensor[i],trans)

    plt.imshow(img_PIL)
    plt.show()
```

![image-20230901120007501](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011200549.png)

![image-20230901120018342](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011200389.png)

![image-20230901120040289](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011200337.png)

![image-20230901120051096](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011200145.png)

![image-20230901120100479](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011201525.png)

> 获得上左，上右，下左，下右，中，5张图。

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.TenCrop(112, vertical_flip=True), # 将获得5张图，是一个tuple元组，要拆开
     transforms.Lambda( lambda crops: torch.stack( [ (transforms.ToTensor()(each)) for each in crops ] ) ),
     ]
)
```

![image-20230901120445513](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011204561.png)

![image-20230901120458593](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011204641.png)

![image-20230901120507112](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011205159.png)

![image-20230901120516782](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011205829.png)

![image-20230901120525068](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011205117.png)

> 再增加5张翻转的图片。

### **8、RandomVerticalFlip & RandomHorizontalFlip**

![image-20230901120659605](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011206642.png)

**依概率水平（左右）或垂直（上下）翻转图片** 

* **P:**翻转概率

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.RandomHorizontalFlip(p=1),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901121032031](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011210171.png)

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.RandomVerticalFlip(p=1),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901121115943](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011211081.png)

### **9、RandomRotation**

![image-20230901121237543](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011212582.png)

**随机旋转图片**

* **degrees ：**旋转角度
  * 当为a时，在(-a, a)之间选择旋转角度
  * 当为(a, b)时，在(a, b)之间选择旋转角度
* **resample :**重采样方法
* **expand :**是否扩大图片,以保持原图信息
* **center :**旋转点设置,默认中心旋转

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.RandomRotation(90),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901121626642](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011216777.png)

> 可以看到当旋转后，四个角被转到外面，如果要保留角部信息，则transforms.RandomRotation(90, expand=True)

![image-20230901121841693](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011218787.png)

### **10、Pad**

![image-20230901122134043](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011221082.png)

**对图片边缘进行填充**

- **padding :**设置填充大小
  - 当为a时，上下左右均填充a个像素
  - 当为(a, b)时,上下填充b个像素，左右填充a个像素 
  - 当为(a, b, c, d)时,左，上，右,下分别填充a, b, c, d
- **pad_if_need:**若图像小于设定size，则padding
- **padding_mode :**填充模式，有4种模式
  - **constant :**像素值由fill设定
  - **edge :**像素值由图像边缘像素决定
  - **reflect :**镜像填充,最后一个像素不镜像
  - **symmetric :**镜像填充,最后一个像素镜像
- **fill :** constant时,设置填充的像素值

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.Pad(padding=26,fill=(255,0,0)),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901122345048](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011223156.png)

### **11、ColorJitter**

![image-20230901122447232](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011224272.png)

**调整亮度、对比度、饱和度和色相**

* **brightness:**亮度调整因子
  * 当为a时,从［max(0, 1-a), 1+a］中随机选择
  * 当为(a, b)时,从［a, b］中随机选择
* **contrast:**对比度参数，同brightness
* **saturation:**饱和度参数，同brightness 
* **hue：**色相参数，
  * 当为a时，从［-a, a］中选择参数，注：0 <= a <= 0.5
  * 当为(a, b)时，从［a, b］中选择参数，注：-0.5 <= a <= b <= 0.5

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.ColorJitter(hue=0.5),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901123008832](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011230986.png)

### **12、Grayscale & RandomGrayscale**

![image-20230901123105073](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011231114.png)

**依概率将图片转换为灰度图**
•	**num_ouput_channels：**输出通道数 只能设1或3
•	**P:**概率值,图像被转換为灰度图的概率

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)
     transforms.Grayscale(num_output_channels=3),

     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

![image-20230901123317094](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011233181.png)

### **13、RandomErasing**

![image-20230901124000103](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011240145.png)

**对图像进行随机遮挡**

- **p:**概率值,执行该操作的概率
- **scale:**遮挡区域的面积
- **ratio:**遮挡区域长宽比
- **value:**设置遮挡区域的像素值,(R, G, B) or (Gray)

```py
trans = transforms.Compose(
    [
     transforms.Resize((224,224)),# 必须是元组(224,224)

     transforms.ToTensor(),
     transforms.RandomErasing(p=1,scale=(0.2,0.3),ratio=(0.2,0.3)),

     # transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]
)
```

> 注意，RandomErasing必须要在tensor下进行，所以之前要加transforms.ToTensor()

![image-20230901124641791](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011246903.png)

### **14、自定义转换**

```py
class Noise_transform():
    def __init__(self, snr, p):
        # 信噪比
        self.snr = snr
        # 中入噪声的概率
        self.p = p

    def __call__(self, img):
        if np.random.uniform(0,1) < self.p:
            noise_p = self.p
            signal_p = 1 - self.p
            img = np.array(img).copy()
            H, W, C = img.shape
            # 按signal_p, noise_p/2, noise_p/2的概率生成0,1,2矩阵
            mask = np.random.choice((0,1,2), size=(H,W,1), p=(signal_p, noise_p/2, noise_p/2))
            mask = np.repeat(mask, C, axis=2)
            img[mask==1] = 0
            img[mask==2] = 255
            return Image.fromarray(img.astype('uint8')).convert('RGB')

        else:
            return img
```

> 这里实现了一个类，对图像加入噪声，通过初始化参数snr, p（信噪比，中入噪声的概率）来初始化该类。

![image-20230901131815181](https://raw.githubusercontent.com/HX-1234/NoteImage/main/202309011318282.png)