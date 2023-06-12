### 模型信息
- Introduction
The Authors present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. The authors motivate the choice of the convolutional architecture via a localized first-order approximation of spectral graph convolutions. The model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that their approach outperforms related methods by a significant margin.

- Paper
[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) 

- 模型代码来源
This case includes code from the MIT License open source project at https://github.com/tkipf/pygcn

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


### 数据集
#### 数据集下载地址
https://github.com/tkipf/pygcn/tree/master/data/cora  

#### 预处理
无需预处理




### 框架与芯片支持情况
|            | Pytorch | Paddle | TensorFlow2 |
| ---------- | ------- | ------ | ----------- |
| Nvidia GPU | ✅       | N/A    | N/A         |
| 昆仑芯 XPU | N/A     | N/A    | N/A         |
| 天数智芯   | N/A     | N/A    | N/A         |