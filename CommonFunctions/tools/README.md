# tools

## tools.commom_tools

### use_svg_display

```python
use_svg_display()
"""使用svg格式在Jupyter中显示绘图。"""
```

### set_figsize

```python
set_figsize(figsize=(3.5, 2.5))
"""设置matplotlib的图表大小。"""
```

### set_axes

```python
set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴。"""
```

### plot

```python
plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点。"""
```

### Timer

```python
class Timer:  # @save
    """记录多次运行时间"""
```

### synthetic_data

```python
synthetic_data(w, b, num_examples):  #@save
    """生成 y = Xw + b + 噪声。"""
```

### linreg

```python
linreg(X, w, b):  #@save
    """线性回归模型。"""
```

### squared_loss

```python
squared_loss(y_hat, y):  #@save
    """均方损失。"""
```

### sgd

```python
sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降。"""
```

### load_array

```python
load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器。"
```

### get_fashion_mnist_labels

```python
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签。"""
```

### show_images

```python
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
```

### load_data_fashion_mnist

```python
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
```

### accuracy

```python
accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
```

### evaluate_accuracy

```python
evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度。"""
```

### Accumulator

```python
class Accumulator:  #@save
    """在`n`个变量上累加。"""
```

### train_epoch_ch3

```python
train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）。"""
```

### Animator

```python
class Animator:  #@save
    """在动画中绘制数据。"""
```

### train_ch3

```python
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）。"""
```

### predict_ch3

```python
predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）。"""
```



