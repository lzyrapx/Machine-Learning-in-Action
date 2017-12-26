# Ch04 朴素贝叶斯(Naive Bayes)

## 朴素贝叶斯概述

朴素贝叶斯是一种简单但是非常强大的线性分类器。它在垃圾邮件分类，疾病诊断中都取得了很大的成功。它只所以称为朴素，是因为它假设特征之间是相互独立的，但是在现实生活中，这种假设基本上是不成立的。那么即使是在假设不成立的条件下，它依然表现的很好，尤其是在小规模样本的情况下。但是，如果每个特征之间有很强的关联性和非线性的分类问题会导致朴素贝叶斯模型有很差的分类效果。

朴素贝叶斯分类器通过求出使得概率 ![\\inline P\(X|W\)](http://latex.codecogs.com/png.latex?%5Cinline%20P%28X|W%29) 最大化的类别 ![\\inline X](http://latex.codecogs.com/png.latex?%5Cinline%20X)，以确定特征向量 ![\\inline W = \(w_1, w_2, w_3, \\dots\)](http://latex.codecogs.com/png.latex?%5Cinline%20W%20%3D%20%28w_1%2C%20w_2%2C%20w_3%2C%20%5Cdots%29) 最有可能属于的类别。

根据条件概率公式，![\\inline P\(X|W\) = \\frac{P\(W|X\) \\times P\(X\)}{P\(W\)}](http://latex.codecogs.com/png.latex?%5Cinline%20P%28X|W%29%20%3D%20%5Cfrac{P%28W|X%29%20%5Ctimes%20P%28X%29}{P%28W%29})。![\\inline P\(X\)](http://latex.codecogs.com/png.latex?%5Cinline%20P%28X%29) 可以视为一个先验概率，用类别 ![\\inline X](http://latex.codecogs.com/png.latex?%5Cinline%20X) 在样本中的频率近似算出。![\\inline P\(W\)](http://latex.codecogs.com/png.latex?%5Cinline%20P%28W%29) 虽然很难计算，但它是一个与 ![\\inline X](http://latex.codecogs.com/png.latex?%5Cinline%20X) 无关的常数，而我们只需要找到使得概率最大化的 ![\\inline X](http://latex.codecogs.com/png.latex?%5Cinline%20X)，只要比较大小，并不需要精确算出这个概率，所以可以无视这个值。

问题就在于如何计算 ![\\inline P\(W|X\)](http://latex.codecogs.com/png.latex?%5Cinline%20P%28W|X%29)，这里就是朴素贝叶斯分类器的“朴素”体现出来的地方。朴素贝叶斯分类器做了一个强假设，认为 ![\\inline W](http://latex.codecogs.com/png.latex?%5Cinline%20W) 里的每个特征都是互相独立的，即 ![\\inline P\(W|X\) = P\(w_1|X\) \\times P\(w_2|X\) \\times P\(w_3|X\)\\dots](http://latex.codecogs.com/png.latex?%5Cinline%20P%28W|X%29%20%3D%20P%28w_1|X%29%20%5Ctimes%20P%28w_2|X%29%20%5Ctimes%20P%28w_3|X%29%5Cdots)，这就方便了我们的概率计算。

为了计算某一个特征的概率 ![\\inline P\(w|X\)](http://latex.codecogs.com/png.latex?%5Cinline%20P%28w|X%29)，如果 ![\\inline w](http://latex.codecogs.com/png.latex?%5Cinline%20w) 的取值是离散的，直接使用古典概型计算即可；如果 ![\\inline w](http://latex.codecogs.com/png.latex?%5Cinline%20w) 的取值是连续的，可以假设 ![\\inline w](http://latex.codecogs.com/png.latex?%5Cinline%20w) 服从正态分布。

太多的小概率乘起来，可能会因为结果太小导致下溢或者得到不正确的答案。解决方法是：可以将概率取对数，这样乘法就变成了加法，取值虽然不相同，但也不影响最终答案。

## 朴素贝叶斯背后的数学原理

### 后验概率(Posterior Probabilities)
### 条件概率(Conditional Probabilities)
### 先验概率(Prior Probabilities)
### 现象概率(Evidence Probabilities)
