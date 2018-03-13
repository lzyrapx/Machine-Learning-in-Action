# Ch15 - 大数据与MapReduce(Big data and MapReduce)

## 总结：

### 当运算需求超出了当前资源的运算能力，可以考虑购买更好的机器，或者租用网络服务并使用MapReduce框架并行执行。另一个情况是，运算需求超出了合理价位下所能购买到的机器的运算能力。其中一个解决方法是将计算转成并行的作业，MapReduce就提供了这种方案的一个具体实施框架。在MapReduce中，作业被分成map阶段和reduce阶段。

### 一个典型的作业流程是先使用map阶段并行处理数据，之后将这些数据在reduce阶段合并。这种多对一的模式很经典，但不是唯一的流程方式。mapper和reducer之间传输数据的形式是key/value对。一般地，map阶段后数据还会按照key值进行排序。Hadoop是一个流行的可行MapReduce作业的java项目，它同时提供非Java作业的运行支持，叫做Hadoop流。

### 很多机器学习算法都可以容易地写成MapReduce作业，而某些需要经过重写和创新性的修改，才能在MapReduce上运行。
