# 基于概率的中文分词算法

## 概述
本算法采用基于ngram的最大概率方法，将句子根据词典构建为一个DAG，并根据条件概率计算分词概率。

## 语言模型

### (尝试)神经语言模型
基于LSTM的神经语言模型，基于TensorFlow官方实现https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb.

由于神经语言模型的几个缺陷，本算法没有采用神经语言模型：
1. 需要数据量大。
2. 计算耗时久。由于未能找到预训练好的中文神经语言模型，需要自己从头训练。实验表明，使用2x NVIDIA GTX 1080Ti训练10 epoch需要~5h。
3. **容易过拟合**。在多次调整dropout等参数后语言模型仍有严重的过拟合现象。

### NGRAM语言模型
本算法最终采用ngram语言模型计算概率。本算法支持n为任意值的ngram模型,可以方便地通过参数调节n的值。最终采用n=2作为最终语言模型，并采用Laplace平滑。

## 数据预处理
### 特殊词语
为了减少特殊词语低概率的问题，将特殊词语统一替换成标识符。替换规则如下:
|   Source   	| Target 	|
|:----------:	|:------:	|
|  英文字符  	|    l   	|
| 阿拉伯数字 	|    d   	|

### 其他预处理
1. 全半角转换. 将全部全角标点统一转换为半角标点(采用`unicodedata`包).
2. 简繁转换. 将繁体字符全服转换为简体字符(采用`opencc`工具).
3. 去除连续字符. 将连续空格、连续英文字符和连续阿拉伯数字去除.

## 使用

本使用方式在Ubuntu 16.04 + Anaconda Python 3.6下测试通过。不保证其他环境的运行效果。

若要运行本程序，Python解释器版本必须高于3.6。

本说明中性能指标均为以下环境中测试得到. 数据预处理过程中可能需要~2 Gb的内存占用.
|  Type  	|                    Device                    	|
|:------:	|:--------------------------------------------:	|
|   CPU  	| 2x Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz 	|
|   GPU  	|         2x NVIDIA GeForce GTX 1080Ti         	|
| Memory 	|                     256G                     	|



## 训练
### 简繁转换
使用[`opencc`](https://github.com/BYVoid/OpenCC)将繁体数据集转化为简体数据集。
命令示例:
```bash
mv cityu_training.utf8 cityu_training.utf8.zh-hk
opencc -i cityu_training.utf8.zh-hk -o cityu_training.utf8 -c t2s.json
```

> 在测试环境上，简繁转换共用时约30分钟.

### 预处理
执行
```bash
python dataset/Preprocess.py <dataset_source> <output_path>
```
参数说明:
- `<dataset_source>`: icwb2 数据集**文件夹**路径, 算法会自动读取路径下训练和测试数据.
- `<output_path>`: 预训练文件输出**文件夹**

> 在测试环境上，预处理用时 < 30s

### 分词
首次运行时，程序会计算训练数据中ngram概率并将其以pickle文件缓存至`graph/cache`文件夹下. 之后每次运行程序会在初始化阶段加载pickle文件.

分词示例:
```python3
dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
corpus_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train"
graph = ProbCalculator(dict_path, corpus_path, ngram_size=2)
g: str = graph.calc("去北京大学玩")

print(g)
```

输出:

```
load saved ngram
load saved word dict
initialization completed.
去 北京大学 玩
```

## Ablation Study

### n-gram中n的选择
在词典较大时，n=2时数据稀疏已经比较明显。对几个错误分词结果进行分析表明，许多错误分词是由于此表里根本没有此搭配。例如`中华人民共和国 今天 成立 啦 !`中，训练语料中并没有`中华人民共和国 今天`的组合。在n>2时，数据稀疏的后果更为严重，导致整个句子中出现大量概率为0的成分。

### 预处理的作用
数据预处理是一种缓解数据稀疏的方法。例如语句`1977年9月`被处理为`d年d月`，可以匹配任意年月的语句。


## Error Study
数据稀疏是分词错误的重要原因。在不采用平滑时，测试语料中有30%的概率出现了概率为0的现象。

## Future Work
1. 可以使用神经语言模型计算概率(可能会导致速度急剧下降)
2. 进一步提高预处理效率. 数据预处理中sklearn.feature_extraction.text.CountVectorizer计算ngram和re.sub替换特殊词语部分占用总预处理时间的60%。可以采用更高效的实现从而提升预处理速度。
3. 采取更加合适的smoothing策略，如Katz Smoothing等
4. 优化代码，采取更高效的数据结构。例如采取Native的数据结构而不是class来提高效率.