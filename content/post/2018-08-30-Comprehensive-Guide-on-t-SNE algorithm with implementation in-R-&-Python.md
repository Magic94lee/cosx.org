title: 在R＆Python中实现t-SNE算法综合指南
author: 
  - 黄理强
date: '2018-08-30'
meta_extra: "原作者：Saurabh.Jaju2；译者：黄理强"
categories:
  - 机器学习
tags:
  - 机器学习
  - 翻译
forum_id: ""


介绍

想想一下，你获得一个包含数百个特征的数据集，并且对数据所属的域几乎一无所知。你需要识别数据中的隐藏规律、探索和分析数据集。不仅如此，你还需发现数据中是否有规律———它是信号还是噪音？

这个想法会让你觉得困难么？当我第一次遇到这个情况时，我感到手心出汗，无从下手。你是否想知道如何探索多维数据集？这是很多数据科学家经常提出的问题之一。在本文中，我将带你领略一种神奇的方法来完成此任务。

什么是PCA?

现在，你们中的一些人会说：“我将利用PCA进行降维和可视化“。你的想法是对的！PCA绝对是降维和可视化多特征数据集的不错选择。但是，如果你能使用比PCA更先进的东西呢？(如果你不了解PCA，我强烈建议你先阅读这篇文章)

如果你可以轻松的搜索非线性样式的数据？在本文中，我将向你介绍一种算法t-SNE(2008)，它比PCA(1933)更加有效。首先我将介绍t-SNE算法的基础知识，然后再向你介绍为什么t-SNE非常适合降维。

同时你也会得到在R&Python中使用t-SNE的实践知识。

继续阅读！

目录

1. 什么是t-SNE?
2. 什么是降维？
3. t-SNE是如何适应降维算法空间的
4. t-SNE的算法细节
   - 算法
   - 时间和空间复杂度
5. t-SNE究竟做了啥
6. 使用案例
7. t-SNE与其他降维算法比较
8. 示例实现
   - R
     - 超参数调整
     - 代码
     - 执行时间
     - 结果解释
   - Python
     - 超参数调整
     - 代码
     - 执行时间
9. 运用t-SNE的场合和时间
   - 数据科学家
   - 机器学习比赛参与者
   - 学生
10. 常见的错误理解



1. 什么是t-SNE

(t-SNE) t-分布邻域嵌入算法是用于探索高维数据的非线性的降维算法。它将高维数据映射成二维或者更高的维度，便于人们观察数据。借助于t-SNE算法，下次你处理高维数据时，你绘制探索数据分析图的次数会变少。





2. 什么是降维

为了理解t-SNE是如何产生作用的，我们应该先理解什么是降维？

简单来说，降维是指利用两个维度或者三个维度来代表多维数据(具有彼此相关的多个特征的数据)的技术。

你们中一些人也许会疑问，我们已经可以使用点图、直方图和箱型图来绘制数据以及利用描述性分析来理解数据中的规律，为何我们需要降维。

因为即使你可以理解数据的规律，利用简单图标来展现数据，对于没有统计背景的人来说理解其含义还是困难的。同时如果数据中存在上百个特征，你必须通过成百上千的图形来理解数据。(了解更多降维)

在降维算法的帮助下，你能明确的展示数据。



3. t-SNE是如何适应降维算法空间的

现在你已经理解什么是降维了。让我们看看如何使用t-SNE算法来降维。

以下是一些降维算法：	

1. PCA (线性)
2. t-SNE (非参数/非线性)
3. Sammon mapping (非线性)
4. Isomap (非线性)
5. LLE (非线性)
6. CCA (非线性)
7. SNE (非线性)
8. MVU (非线性)
9. Laplacian Eigenmaps (非线性)

好消息是你只需要研究上述的两种算法，即可有效地可视化低维数据的算法—PCA和t-SNE。

PCA的限制

PCA是线性算法。它不能有效地解释特征之间复杂多项式关系。另一方面，t-SNE基于邻域图上随机游走的概率分布来发现数据内的结构。

线性降维算法有一个较大的问题是，它们专注于在较低的维度表示中将不相似的数据分离开来。但是为了能在低维上展示高维数据，相似的数据点也应该在低维上有相近的表示，这不是线性降维算法能做的。

现在我们对于PCA能做的事情已经有了简单的认识。

局部方法将流形上的相邻点映射到低维表示空间的相邻点。另一方面，全局方法试图在所有尺度上保持几何形状，即将临近点映射为临近点，将分离点映射为远离点。

重要的是除了t-SNE之外的大多数非线性降维算法不能同时保留数据的局部和全局结构。



4. t-SNE的算法细节(选读)

这部分适用于对深入了解算法感兴趣的人。如果你不想详细了解算法的数学细节，可以跳过。

让我们理解你为什么需要了解t-SNE以及t-SNE的算法细节。t-SNE是邻域嵌入算法(SNE)算法的改进版本。

4.1 算法

步骤 1

邻域嵌入算法(SNE)第一步将数据点之间的高纬度欧几里得距离转变为代表相似性的条件概率。数据点与数据点之间的相似性是条件概率，将会挑选为相近点，如果相近点的挑选与以为中心的高斯分布下概率密度成正比。

对于相邻点，相对较高，对于相距甚远的数据点，几乎是无穷小的(对于合理的高斯方差值，)。在数学上，条件概率由下列式子给出：



其中，是已数据点为中心的高斯分布的方差。

如果你对数学没有兴趣，以这种方式思考，该算法首先将最短距离(直线)转变成点与点之间的相似概率。点与点之间的相似概率是：将会挑选为相近点的条件概率，如果相近点的挑选与以为中心的高斯分布下概率密度成正比。

步骤 2

对于高维数据点和的低纬度的对应点和。我们可以计算出相似条件概率，表示为。



记住，$p_{i|i}$和$p_{j|j}$设置为0，我们只想模拟一对数据点的相似性。

简单来说，步骤1和步骤2计算了一对点之间相似性的条件概率在

	1.高维空间

	2.低维空间

为了简单起见，试着在细节上理解这一点。

让我们将三维空间映射为二维空间。步骤1和步骤2所做的是计算在三维空间中点与点之间的相似性的概率以及计算在二维空间中相应点之间的相似性的可能性。从逻辑上来说，条件概率和必须相等才能完美的表示数据点在不同维度空间的相似性，即对于不同维度空间同一数据点的和必须相等。

通过这个逻辑，SNE尝试最小化条件概率的差异。

步骤 3



下面说的是SNE与t-SNE的区别。

为了衡量不同表示空间中点与点之间的条件概率差别的总和的最小值，SNE运用梯度下降的方法计算了全部数据点之间的 Kullback-Leibler分歧总和的最小值。我们得了解KL分歧是非对称的。

换句话说，SNE的损失函数关注于在映射中保持数据的局部结构(为了保持在高维空间中高斯方差的合理值，)。

另外，优化损失函数是困难的，由于计算的低效。

因此，t-SNE也是尝试最小化条件概率的差值的总和。但是它使用的是SNE损失函数的对称形式，并且带有简单的梯度。同时，t-SNE在低维空间运用了厚尾分布来缓解拥挤问题(相比较于能够容纳相邻数据点的区域，在二维空间映射的能够容纳中距离数据点的区域是不够大的)以及SNE的优化问题。

步骤 4

如果我们看到计算条件概率的可能性的公式，我们一直忽略了讨论的差异。被选择留下的参数是建立在每个高维数据点的t-分布的方差。对于数据集中的所有数据点，存在一个最佳的值是不太可能的，因为数据的密度是多样的。相较于稀疏的区域，密集的区域更适合较小的值。任何特定的值都会在其他的数据点上导致概率分布。

该分布的熵随着增长而增长。t-SNE使用二分搜索来获得的值，在这个过程中，产生了带有由确定的复杂度的。复杂度的定义如下：

其中，H()指的香农信息熵 (以比特为单位)



复杂度可以解释为邻居的有效数量的平滑度量。对于复杂度的变化SNE的表现显得比较稳健，典型值在5和50之间。

损失函数的最小值通过梯度下降实现。在物理上，梯度可以解释为映射点和其他映射点之间产生的一系列作用力所产生的合力。所有的作用力沿着(- )方向施加力。和之间的作用力吸引或者排斥映射点，基于映射点之间的距离是否过大或者过小以至于无法表示两个高维数据点之间的相似性。和之间的作用力的大小与他们之间的长度以及刚度成正比。刚度是指数据点与映射点之间的成对不相似程度(pj|i – qj|i + p i| j − q i| j )[1]。

4.2 时间空间复杂度

现在我们已经理解了算法，让我们看算法具体的表现。你也许已经观察到，算法计算成对条件概率并且尝试着将高维表示空间与低维表示空间的概率的差值的总和最小化。这涉及大量的计算。所以该算法在系统资源方面十分繁重。

t-SNE具有基于样本数的二次时间复杂度和二次空间复杂度。 当该算法运用于包含10,000多个观测值的数据集时，会造成速度变慢以及计算资源枯竭。

5. t-SNE 发挥的作用

在我们了解该算法是如何工作的数学描述之后，我们总结一下，我们学到了什么。以下是t-SNE如何工作的简要说明。

实际上非常简单，t-SNE是非线性降维算法，它通过多维度数据点的相似性来识别观察到的簇从而发现数据的规律。但是 它不是聚类算法，它是降维算法。因为它将多维度数据映射到较低的维度空间，输入的特征变得不可再识别。因此,你不能仅仅基于t-SNE的输出进行任何推断。所以本质上来说，t-SNE仅仅是数据探索和可视化技术。

但是t-SNE可以用于分类和归类过程，将其输出的特征当做其他分类算法的输入特征。

6. 使用案例

你可能会提问该算法的运用场景是什么。t-SNE能运用于几乎所用的高维数据集。它广泛运用于图像处理、NLP、基因组数据和语音处理。它已经被用于改进大脑和心脏扫描的分析。以下是一些例子：

6.1 面部表情识别

在FER(面部表情识别)上已经取得了很多进展，并且对于FER研究了很多算法比如PCA。但是，由于降维以及分类的困难，FER依然是个挑战。t-分布邻域嵌入算法(t-SNE) 将高维数据转变为相关的低维次空间，同时利用其他算法比如：AdaBoostM2，随机森林，逻辑回归，NNs以及其他算法作为表情分类的多分类器。

在一种基于日本女性面部表情数据的面部识别尝试中，利用t-SNE和AdaBoostM2算法获得的结果显示，相比较于传统算法，例如PCA、LDA、LLE和SNE，t-SNE在FER(面部表情识别)标新更好。[2]

在数据上实现算法组合的流程图如下：

预处理 → 归一化 → t-SNE→分类算法

                      PCA      LDA    LLE     SNE    t-SNE

SVM               73.5%  74.3%  84.7%  89.6%  90.3%

AdaboostM2   75.4%  75.9%  87.7%  90.6%  94.5%

6.2 识别肿瘤亚群（医学影像）

质谱成像(MSI)是一种同时直接从组织提供数百种生物分子的空间分布的技术。空间映射的t-分布邻域嵌入算法(t-SNE) 是一种将数据非线性可视化的算法，能够更好地解决生物分子肿瘤内异质性。

以无偏的方式，t-SNE可以揭示肿瘤亚群，其与胃癌患者存活率和乳腺癌原发性肿瘤中的转移状态在统计学上相关。对每个t-SNE集群进行残存分析将提供非常有用的结果。[3]

6.3 使用wordvec进行文本比较

词向量表示着捕获许多语言属性，如性别、时态、多元甚至语义概念，例如“首都”。利用降维，可以计算出一个2D映射，在这之中语义相近的单词彼此靠近。这种技术的组合可用于提供不同文本源的鸟瞰视图，包括文本摘要以及源材料。这使得用户可以像地理地图一样探索文本源。[4]

7.  t-SNE与其他降维算法相比较

当t-SNE与其他算法比较性能时，我们将基于准确度而不是时间与资源要求来比较	t-SNE与其他算法。

与PCA和其他线性降维模型相比，t-SNE的输出提供更好地结果。这是因为诸如经典缩放的线性方法不能很好的构建曲线流形。它侧重于保留广泛分离的数据点之间的距离，而不是保留附近数据点之间的距离。

t-SNE通过在高维空间中使用的高斯核定义了数据的局部和全局结构之间的软边界。对于那些高斯标准偏差相近的数据点组，对于其距离建模的重要性几乎与这些大小的大小无关。此外，t-SNE基于数据的局部密度(通过强制每个条件概率分布具有相同的复杂度)确定了每个数据点的局部邻居大小。这是因为算法在数据的全局结构和局部结构之间定义了软间隔。与其他非线性降维算法不同，t-SNE的性能优于其他算法。

8. 示例

让我们在MNIST手写数字集上运用t-SNE算法。这是运用图像处理最受欢迎的数据集之一。

1. 在R中使用

利用“Rtsne”包，可以实现在R中运用t-SNE的。可以在R控制台中键入的以下命令将“Rtsne”软件包安装在R中：

    install.packages(“Rtsne”)

- 超参数调整





- 代码

MNIST数据集可以从MNIST网站下载，可以利用少量的代码将数据集转换成csv文件。举个例子，请下载以下预处理的MNIST数据。 链接

    ## calling the installed package
    train<- read.csv(file.choose()) ## Choose the train.csv file downloaded from the link above  
    library(Rtsne)
    ## Curating the database for analysis with both t-SNE and PCA
    Labels<-train$label
    train$label<-as.factor(train$label)
    ## for plotting
    colors = rainbow(length(unique(train$label)))
    names(colors) = unique(train$label)
    
    ## Executing the algorithm on curated data
    tsne <- Rtsne(train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
    exeTimeTsne<- system.time(Rtsne(train[,-1], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500))
    
    ## Plotting
    plot(tsne$Y, t='n', main="tsne")
    text(tsne$Y, labels=train$label, col=colors[train$label])

- 实施时间

    exeTimeTsne
      user       system  elapsed 
      118.037   0.000  118.006
    
    exectutiontimePCA
       user     system   elapsed 
      11.259   0.012     11.360

从上面可以看出，与PCA相比，在相同的数据集上t-SNE执行需要更长的时间。

- 结果展示

图形可以用于数据探索分析。算法的输出值x与y轴的坐标以及损失值能作为分类算法的特征。





2. 在Python中使用

提醒一点，“pip install tsne”会产生错误。不推荐安装“tsne”包，可以从sklearn包中获得t-SNE算法。

- 超参数调整



- 代码

下面的代码取自sklearn网站的sklearn示例。

    ## importing the required packages
    from time import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import offsetbox
    from sklearn import (manifold, datasets, decomposition, ensemble,
                 discriminant_analysis, random_projection)
    ## Loading and curating the data
    digits = datasets.load_digits(n_class=10)
    X = digits.data
    y = digits.target
    n_samples, n_features = X.shape
    n_neighbors = 30
    ## Function to Scale and visualize the embedding vectors
    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)     
        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        if hasattr(offsetbox, 'AnnotationBbox'):
            ## only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(digits.data.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    ## don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
    
    #----------------------------------------------------------------------
    ## Plot images of the digits
    n_img_per_row = 20
    img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 10 * i + 1
        for j in range(n_img_per_row):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('A selection from the 64-dimensional digits dataset')
    ## Computing PCA
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca,
                   "Principal Components projection of the digits (time %.2fs)" %
                   (time() - t0))
    ## Computing t-SNE
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()

- 运行时间

    Tsne: 13.40 s
    
    PCA: 0.01 s







9. 何时何地运用t-SNE？

9.1 数据科学

对于数据科学家来说，使用t-SNE遇到的主要问题是算法黑盒子一般的本质。这也阻断了根据结果提供推论和见解的过程。此外，该算法的另一个问题是在连续的运行中并不能得出一个相似的结果。

那么你如何使用这个算法呢？使用该算法的最好方式是用于探索性数据分析。这会让你了解数据中隐藏的规律。同时它还可以作为其他分类和聚类算法的输入参数。

9.2 机器学习高手

将数据降维至二维或者三维，并用非线性学习器学习数据。使用一系列学习器的堆积于集合。然后，你可以使用XGboost来增强t-SNE来获得更好地结果。

9.3 数据科学爱好者

对于刚刚接触数据科学的数据科学爱好者来说，该算法在研究和性能增强方面提供了更好地机会。已经有一些研究论文试图利用线性函数来改善算法的时间复杂度。但是仍需要最佳的解决方案。关于将t-SNE运用于各种NLP问题和图像处理应用的研究论文是一个未开发的领域。



10. 常见的谬论

以下是在解释t-SNE结果时要避免的一些常见错误：

1. 为了使算法正确执行，复杂度应该小于点的数量。同时，建议复杂点设置在(5,50)的范围之内。
2. 有时，具有同样超参数的不同运行可能产生不同的结果。
3. 任何t-SNE图中的簇大小都不能用标准偏差，色散或者其他类似的测量方式进行评估。这是因为t-SNE拓展了更密集的集群，并将稀疏的集群收缩到均匀的集群大小。这是它产生清晰的图像的原因。
4. 簇之间的距离可能改变因为整体几何与最优复杂度密切相关。在具有许多含有不同元素数量的数据簇的数据集中，一个复杂度不能优化所有聚类的距离。
5. 同时，能发现数据集中的噪音的规律。在确定数据中存在的规律之前，必须多次运行具有不同超参数的算法。
6. 在不同的复杂度下，可以观察到不同的簇形状。
7. 无法基于t-SNE图进行拓扑，在进行任何评估之前必须观察多个图。



引用

[1] L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using  t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008

[2] Jizheng Yi et.al. Facial expression recognition Based on t-SNE and AdaBoostM2.

IEEE International Conference on Green Computing and Communications and IEEE Internet of Things and IEEE Cyber,Physical and Social Computing (2013)

[3]  Walid M. Abdelmoulaa et.al. Data-driven identification of prognostic tumor subpopulations using spatially mapped t-SNE of mass spectrometry imaging data.

12244–12249 | PNAS | October 25, 2016 | vol. 113 | no. 43

[4]  Hendrik Heuer. Text comparison using word vector representations and dimensionality reduction. 8th EUR. CONF. ON PYTHON IN SCIENCE (EUROSCIPY 2015)



后记

我希望你阅读此文章有所收获。在本文中，我尝试从各个方面帮助你学习t-SNE。我确信你会激动地探索t-SNE算法并且运用它。

要是你觉得它优于PCA，分享你运用t-SNE算法的经历。如果你有任何疑问或问题，请随时在评论部分发表意见。
