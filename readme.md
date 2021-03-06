https://www.datafountain.cn/competitions/518

# 运行
运行baseline.py前请运行data文件夹下的dataset.py

##### Linux

在当前目录运行一个脚本示例如下

```
CUDA_VISIBLE_DEVICES=0 python baseline.py --bert_id 0
```
##### Windows

```
set CUDA_VISIBLE_DEVICES=0
python baseline.py --bert_id 0 
```

不同的bert_id指定选择不同的tokenizer的ckpt版本，可以参考[baseline.py](./baseline.py#L21)查看id对应的ckpt

CUDA_VISIBLE_DEVICES根据设备上的NVIDIA GPU个数及对应序号可指定不同的值

# dataset

比赛提供的数据集分别是`train.csv`和`test.csv`，要对数据进行预处理则运行`dataset.py`。提交的文件夹中已经完成数据集的预处理，不需要再运行。

测试集的量比`submit_example.tsv`少了一行，我们生成文件时为了格式方便是直接复制`submit_example.tsv`来修改的，所以我们的`submit_example.tsv`删掉了最后一行，如果使用比赛网站下载的原数据测试运行的话也请删掉一行数据。

## batch_size&EPOCHS

由于显卡性能限制，batch_size设置为1（要占8G不到一点的专用显存），如果想要加快训练速度的话可以在显卡承受范围内上调。

迭代轮数EPOCHS也可以调整。

## 其它

如果提示缺少某文件或文件不存在则创建新的对应名字的空文件/文件路径即可，可能是跟环境下程序创建文件的权限有关。

如果想要完全从头测一遍建议直接运行，提交的文件夹中各文件/文件路径都已经生成好，文件会自动覆写，以免出现上述错误。

