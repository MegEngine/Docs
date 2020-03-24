.. _load_pytorch:

在 MegEngine 中嵌入 PyTorch 子图（Experimental）
===================================================

MegEngine 支持在网络搭建过程中嵌入 PyTorch 模块。
该功能可以方便用户轻松地将已有的 PyTorch 模块移植到 MegEngine 框架中使用。

安装本章节所需的 Python 库

.. code-block:: bash
    
    pip install torch torchvision ninja --user

对于一个已有的 PyTorch 模块，我们可以利用 MegEngine 中提供的 :class:`~.pytorch.PyTorchModule` 将它包裹（wrap）成与 MegEngine :class:`~.Module` 兼容的模块。

为了方便演示，假设有一个现成的基于 PyTorch 实现的特征提取模块 ``LeNetExtractor`` （不包含 LeNet 网络结构中的分类层）。在 MegEngine 框架中，我们将这个 PyTorch 模块包裹，只需额外实现一层线性分类器，即可完成 LeNet 网络的搭建。

代码如下：

.. code-block::
    
    import megengine.module as M 
    from megengine.module.pytorch import PyTorchModule
    from megengine.core.graph import get_default_device

    class LeNet(M.Module):
        def __init__(self, lenet_extractor):
            super(LeNet, self).__init__()
            # 将其包裹
            self.lenet_extractor_wrap = PyTorchModule(lenet_extractor, get_default_device())
            # 用 MegEngine 搭一个线性分类器
            self.mge_classifier = M.Linear(84, 10)

        def forward(self, x):
            x = self.lenet_extractor_wrap(x)
            x = self.mge_classifier(x)
            return x

    # 假设我们已经有了 lenet_extractor
    lenet = LeNet(lenet_extractor)

    # 网络训练和测试代码省略
    # ...


基于 PyTorch 的 LeNetExtractor 代码如下：

.. code-block::

    import torch
    import torch.nn as nn

    # 创建一个 PyTorch 版的 LeNet 特征提取模块
    class LeNet_Extract(nn.Module):
        def __init__(self):
            super(LeNet_Torch_Extract, self).__init__()
            # 单信道图片, 两层  5x5 卷积 + ReLU + 池化
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            # 两层全连接 + ReLU
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(120, 84)
            self.relu4 = nn.ReLU()

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            # 拉平 [C, H, W] 三个维度
            x = x.view(-1, 16*5*5)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            return x


