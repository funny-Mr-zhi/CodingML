# Batching

## Theory: Batching V.S. Traditional Methods

### Limitation of traditional Methods
`Batch`: 一批，分批处理。

数据的组织方式其实是和训练过程密切相关的：`train`过程本质上是基于梯度下降算法进行参数优化，每次优化过程通过输入数据来计算梯度，而数据中包含多个样本点，每次输入样本的数量就是Batch操作所关注的核心概念。

传统方法有两种：
* **单样本处理**：对应的优化方法为`Stochastic Gradient Descent`-`SDG`的原始形式。每次只取一个样本 $(x_i, y_i)$用来计算梯度 $\Delta L_i(\theta)$，更新参数。
    * 缺点一：更新方差极大，训练路径不稳定，收敛慢且震荡严重
    * 缺点二：可并行性差，硬件未被充分利用
* **全批次处理**：对应优化方法为`Batch Gradient Descent`-`BGD`。每次更新参数使用整个数据集，计算平均梯度 $\Delta L(\theta) = \frac{1}{N}\sum^n_{i=1}\Delta L_i(\theta)$
    * 缺点：对于现代大型数据集，内存/现存无法容纳

### Mini-Batching
于是基于对现代计算硬件特性和优化算法的考量，引入`mini-batching`概念。

* **硬件角度**：GPU特性
    * GPU采用`SIMD(Single Instruction, Multiple Data)`架构，擅长对不同的数据进行相同的并行化操作。
    * Batching讲B个独立输入样本组织成一个更高维度的输入张量，模型所有的操作都变成了对这个大张量的操作，GPU提供针对这种`张量-矩阵运算`的优化，使其能最大限度利用硬件的并行带宽。
* **优化稳定性**：梯度下降算法（SDG/Adam）
    * 梯度用`mini-batch`内所有的样本计算得到的梯度的平均值来估计
    * 显著降低了梯度的方差
    * `Batch-size`的选择是在偏差和估计之间找到一个平衡，尽管`batch-size`增大会降低方差，但是它可能使模型收敛到一个更“尖锐”的局部最小值。
* **内存限制**

> Q: 理想情况下，Mini-Batch 梯度是全量梯度的无偏估计? 为什么Batch_size变大会影响偏差？

## Frame: How Batching work

本段主要分析`Pytorch`与`PYG/DGL`如何实现`mini-batching`

### Pytorch - 通用框架中的Batching

在`Pytorch`中，数据处理流程被分离成两个主要部分：`Dataset`负责索引并获取单个样本，`DataLoader`负责将这些样本汇聚成批次并高效地送入模型。

这两个类型都是`顶层类/基类`（没有显示继承其它类型）。

#### Dataset

`torch.utils.data.Dataset`是一个抽象类，是所有`Pytorch`数据集的基石，本质上是将数据集是为一个可索引的集合。

通常通过继承`dataset`类来自定义一个数据集，继承时需要实现两个基本魔术方法：`__len__(self)`和`__getitem__(self, index)`，即提供获取长度和通过索引获得指定样本的方法。

#### DataLoader

`torch.uitls.data.DataLoader`位于`Dataset`之上，其作用是将单个的样本组合成批次，并以`可迭代`的形式高效地提供给训练循环。（迭代的优势：一次只加载一个`batch`，用完就扔，避免一次加载所有数据到内存）

**基本调用**
```python
data_loader = Dataloader(
    dataset,        # 传入要加载的Dataset实例
    batch_size,     # 定义每个批次中包含的样本数量B
    shuffle,        # 布尔值，定义是否在每个epoch开始时打乱数据集索引顺序。打乱有助于提高泛化能力
    num_workers,    # 定义用于加载数据集的子进程数，多进程可并行指定Dataset.__getitem__
    collate_fn      # 将多个样本合并为一个Batch张量函数
)
```

**工作流程**
1. 搜集样本：通过调用`Dataset.__getitem__`B次得到一个批次的数据`[(x_1, y_1), (x_2, y_2), ..., (x_B, y_B)]`
2. 合并张量调用`collate_fn`将B个独立样本进行堆叠，将其转换成一个单一的张量

**Collate_fn**
该函数是`Datalaoder`的核心。其默认行为是沿着一个新的维度将所有样本进行堆叠。当Batch中的样本长度不一致时，需要自定义该函数以实现合并功能。

### PYG/DGL - 图Batching

数据结构上，图由点集合和边集合构成，且点和边的数量不固定。

在进行拼接时，图就无法使用默认的`collate_fn`进行拼接，通常采用定制的拼接函数，`PYG`和`DGL`提供了各自的方式。

#### PYG：巨型图策略

`torch_geometric.data.Batch`

* 节点拼接：沿着节点第零个维度进行简单的纵向拼接，形状为$(N_{total}, D)$
* 边拼接：`Edge_index`偏移累加后拼接，形状为$(E_{total}, 2)$, 偏移量 $O_i = \sum_{j=1}^{i-1}|V_j|$
* 图索引：创建额外张量，形状为$(N_{total}, )$，用于记录节点与图的所属关系。

#### DGL：相似但封装更深

`dgl.batch`

## Analysis of Classic Code

调用实验：见`AnalysisOfClassicCode`

### torch.utils.data.dataset

基础的通用Dataset类比较简单，只需要提供两个方法：
* 获取数据集长度：`__len__(self)`，通过`len(dataset)`调用
* 根据索引获取指定样本：`__getitem(self, idx)`，通过`dataset[idx]`调用

### torch.uilts.data.dataloader

创建DataLoader后循环调用，结果见下方代码块。

1. 在一次基本的DataLoader循环遍历数据集时，用到了`dataset`中定义的两个函数`__getitem__`和`__len__`。可见这两个函数在`dataset`类中是必须定义的。
2. 基本的`img_dataset`样本点是形状相同的`torch`类型，在通过默认的`collate_fn`拼接后只是多了一个维度。

```python
"""
输出1：
total batches: 3, total len: 10
using default collate_fn, batch_size=3, shuffle=True, drop_last=True
        Calling __len__
        Calling __len__
        Calling __len__
        Calling __len__
        Calling __getitem__ for index: 9
        Calling __getitem__ for index: 3
        Calling __getitem__ for index: 1
Batch 0: tensor([9, 3, 1])
        Calling __getitem__ for index: 0
        Calling __getitem__ for index: 6
        Calling __getitem__ for index: 2
Batch 1: tensor([0, 6, 2])
        Calling __getitem__ for index: 8
        Calling __getitem__ for index: 5
        Calling __getitem__ for index: 4
Batch 2: tensor([8, 5, 4])
        Calling __len__

输出2：
 Img dataloader
total batches: 2, total len: 10
using default collate_fn, batch_size=4, shuffle=False, drop_last=False
single data shape torch.Size([3, 224, 224])
Batch 0: torch.Size([4, 3, 224, 224])
Batch 1: torch.Size([4, 3, 224, 224])
Batch 2: torch.Size([2, 3, 224, 224])
"""
```


