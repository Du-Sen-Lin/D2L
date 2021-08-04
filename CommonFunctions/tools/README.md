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

### evaluate_loss

```python
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失。"""
```

### download

```python
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名。"""
```

### download_extract

```python
def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件。"""
```

### download_all

```python
def download_all():  #@save
    """下载DATA_HUB中的所有文件。"""
```

### try_gpu

```python
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
```

### try_all_gpus

```python
def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
```

### corr2d

```python
def corr2d(X, K):  #@save
    """计算二维互相关运算。"""
```

### evaluate_accuracy_gpu

```python
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """使用GPU计算模型在数据集上的精度。"""
```

### train_ch6

```python
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
```

### Residual

```python
class Residual(nn.Module):  # @save
    """残差块"""
```



## tools.sequence_tools

### read_time_machine

```python
def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
```

### tokenize

```python

def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符标记。"""
```



### count_corpus

```python
def count_corpus(tokens):  #@save
    """统计标记的频率。"""
```



### Vocab

```python

class Vocab:  #@save
    """文本词表"""
```



### load_corpus_time_machine

```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的标记索引列表和词汇表。"""
```



### seq_data_iter_random

```python
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列。"""
```



### seq_data_iter_sequential

```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列。"""
```



### SeqDataLoader

```python

class SeqDataLoader:  # @save
    """加载序列数据的迭代器"""
```



### load_data_time_machine

```python

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):  # @save
    """返回时光机器数据集的迭代器和词汇表"""
```



### RNNModelScratch

```python
class RNNModelScratch:  #@save
    """从零开始实现的循环神经网络模型"""
```



### predict_ch8

```python

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在`prefix`后面生成新字符。"""
```



### grad_clipping

```python
def grad_clipping(net, theta):  #@save
    """裁剪梯度。"""
```



### train_epoch_ch8

```python
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期（定义见第8章）。"""
```



### train_ch8

```python
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）。"""
```



### RNNModel

```python
#@save
class RNNModel(nn.Module):
    """循环神经网络模型。"""
```



### read_data_nmt

```python
#@save
def read_data_nmt():
    """载入“英语－法语”数据集。"""
```



### preprocess_nmt

```python
#@save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集。"""
```



### tokenize_nmt

```python
#@save
def tokenize_nmt(text, num_examples=None):
    """标记化“英语－法语”数据数据集。"""
```



### truncate_pad

```python
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
```



### build_array_nmt

```python
#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量。"""
```



### load_data_nmt

```python
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词汇表。"""
```



### Encoder

```python
#@save
class Encoder(nn.Module):
    """编码器-解码器结构的基本编码器接口。"""
```



### Decoder

```python
#@save
class Decoder(nn.Module):
    """编码器-解码器结构的基本解码器接口。"""
```



### EncoderDecoder

```python
#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器结构的基类。"""
```



## tools.attention_tools

### show_heatmaps

```python
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """Show heatmaps of matrices."""
```

