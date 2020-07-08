import json

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.optimizer as optim
from megengine.jit import trace

# 使用GPU运行这个例子
assert mge.is_cuda_available(), "Please run with GPU"
# 我们从 megengine hub 中加载一个 resnet50 模型。
resnet = hub.load("megengine/models", "resnet50")

optimizer = optim.SGD(resnet.parameters(), lr=0.1,)

# profiling=True 收集性能数据
@trace(symbolic=True, profiling=True)
def train_func(data, label, *, net, optimizer):
    pred = net(data)
    loss = F.cross_entropy_with_softmax(pred, label)
    optimizer.backward(loss)


resnet.train()
batch_size = 64

# 运行 10 次，保存最后一次的性能结果
for i in range(10):
    batch_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    batch_label = np.random.randint(1000, size=(batch_size,)).astype(np.int32)
    optimizer.zero_grad()
    train_func(batch_data, batch_label, net=resnet, optimizer=optimizer)
    optimizer.step()

# 得到性能数据
prof_result = train_func.get_profile()

# 保存结果为 JSON 格式
with open("profiling.json", "w") as fout:
    json.dump(prof_result, fout, indent=2)
