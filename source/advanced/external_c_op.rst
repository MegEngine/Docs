.. _external_c_op:

自定义 C++ 算子
==============================

本章我们将介绍如何在 MegEngine 中增加自定义的 C++ 算子。
我们实现一个 ``c = a + k * b`` 的算子，其中 ``a`` 、 ``b`` 和 ``c`` 为 :class:`~.megengine.core.tensor.Tensor` 类型，
``k`` 为 float 类型。

C++ 实现
------------------------------

我们在 C++ 中实现函数：

.. code-block:: cpp

    void scaled_add(uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, float k);

``scaled_add`` 的实际计算使用 CUDA 实现。
最后我们使用 `pybind11 <https://github.com/pybind/pybind11>`_ 封装 ``scaled_add`` ，方便在 Python 中调用这个函数。
具体实现如下：

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    // megbrain_pubapi.h 定义了 MegEngine C++ 算子可使用的公共接口
    #include "megbrain_pubapi.h"

    const int threadsPerBlock = 64;

    // CUDA kernel
    __global__ void scaled_add_kern(float* a, float* b, float* c, float k,
                                    size_t N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
            c[i] = a[i] + k * b[i];
    }

    using Tensor = mgb::pubapi::DeviceTensor;
    void scaled_add(uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, float k) {

        // mgb::pubapi::as_versioned_obj<Tensor> 转换 uintptr_t 为 C++ Tensor 指针
        Tensor* a = mgb::pubapi::as_versioned_obj<Tensor>(a_ptr);
        Tensor* b = mgb::pubapi::as_versioned_obj<Tensor>(b_ptr);
        Tensor* c = mgb::pubapi::as_versioned_obj<Tensor>(c_ptr);

        // a->desc.dev_ptr 得到 Tensor 在 device 上的地址
        float* a_dev_ptr = static_cast<float*>(a->desc.dev_ptr);
        float* b_dev_ptr = static_cast<float*>(b->desc.dev_ptr);
        float* c_dev_ptr = static_cast<float*>(c->desc.dev_ptr);

        // a 的 shape 维度
        size_t ndim = a->desc.ndim;

        // a 的长度，假设a非空
        size_t len = 1;
        for (size_t i = 0; i < ndim; i++)
            len *= a->desc.shape[i];

        // 使用输出 Tensor 对应的 cuda stream 来做计算
        auto cuda_stream = static_cast<cudaStream_t>(c->desc.cuda_ctx.stream);

        int blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
        // 调用 CUDA kernel 完成计算
        scaled_add_kern<<<blocks, threadsPerBlock, 0, cuda_stream>>>(
                a_dev_ptr, b_dev_ptr, c_dev_ptr, k, len);
    }

    // 使用pybind11来定义可被 Python 调用的 scaled_add 的接口
    PYBIND11_MODULE(scaled_add, m) {
        m.doc() = "MegEngine external c operator example";
        m.def("scaled_add", &scaled_add, "Calculate c = a + b * k ");
    }


编译动态库
------------------------------

我们保存 C++ 实现为文件 ``scaled_add.cu`` ，然后编译得到动态库 ``scaled_add.so`` 。

编译需要预先安装 `pybind11 <https://github.com/pybind/pybind11>`_ 。
在 Linux 下，可使用以下命令安装：

.. code-block:: bash

 pip3 install pybind11 --user

编译动态库只需要额外传入的 MegEngine 的头文件位置：

.. code-block:: bash

    # MGE_ROOT 是 MegEngine 的安装目录
    MGE_ROOT=`python3 -c "import os; \
                          import megengine; \
                          print(os.path.dirname(megengine.__file__))"`

    # $MGE_ROOT/_internal/include/ 为 MegEngine C++ 头文件目录
    nvcc -Xcompiler  "-fno-strict-aliasing -fPIC -O3" -shared \
         -I$MGE_ROOT/_internal/include/  \
         `python3 -m pybind11 --includes` \
         scaled_add.cu -o scaled_add.so

定义 Python ``functional`` 函数
----------------------------------------------

我们首先实现一个 ``CraniotomeBase`` 的子类 ``ScaledAdd`` ，
在其中定义算子的输入和输出 Tensor 个数，
通过调用在 C++ ``scaled_add`` 来实现前向计算。
如果我们需要在反向传播中使用算子，也可以用类似的方法定义它的梯度计算。
最后我们使用 ``@wrap_io_tensor`` 函数修饰符来定义 ``functional`` 函数 ``scaled_add_external`` 。

.. code-block::

    from megengine._internal.craniotome import CraniotomeBase
    from megengine.core.tensor import wrap_io_tensor

    # 从动态库导入 scaled_add
    from scaled_add import scaled_add


    class ScaledAdd(CraniotomeBase):

        # 定义 ScaledAdd 的输入 tensor 数目
        __nr_inputs__ = 2

        # 定义 ScaledAdd 的输出 tensor 数目
        __nr_outputs__ = 1

        # 非 tensor 参数在 setup 中传入
        def setup(self, k):
            self._k = float(k)

        # 定义前向计算
        def execute(self, inputs, outputs):
            a, b = inputs
            c = outputs[0]

            # 使用 pubapi_dev_tensor_ptr 得到 tensor 第一个元素的地址
            a_ptr = a.pubapi_dev_tensor_ptr
            b_ptr = b.pubapi_dev_tensor_ptr
            c_ptr = c.pubapi_dev_tensor_ptr

            # 调用 C++ 实现的scaled_add
            scaled_add(a_ptr, b_ptr, c_ptr, self._k)

        # 静态图下使用 infer_shape 推导输出的 tensor shape
        def infer_shape(self, inp_shapes):
            # inp_shapes[0] 和 inp_shapes[1] 分别对应输入 a 和 b 的 shape
            assert inp_shapes[0] == inp_shapes[1]
            return [inp_shapes[0]]

        # 定义算子的梯度，此处用 Python 示意实现，我们可以参考 execute 自定义 CUDA 实现
        def grad(self, wrt_idx, inputs, outputs, out_grad):
            assert (
                len(inputs) == 2
                and len(outputs) == 1
                and len(out_grad) == 1
                and wrt_idx in [0, 1]
            )

            # a + b * k 对 a 的导数
            if wrt_idx == 0:
                return out_grad[0]

            # a + b * k 对 b 的导数
            if wrt_idx == 1:
                return out_grad[0] * self._k

    # 定义类似 megengine.functional 的函数
    @wrap_io_tensor
    def scaled_add_external(a, b, k):
        # tensor 类型使用positional 参数，其他类型使用key argument传入
        c = ScaledAdd.make(a, b, k=k)
        return c

之后，我们就可以在 MegEngine 中使用 ``scaled_add_external`` ：

.. code-block::

    import numpy as np
    import megengine as mge

    assert mge.is_cuda_available(), "scaled_add implemented only for CUDA"

    a_val = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    b_val = np.array([4.0, 5.0, 6.0]).astype(np.float32)
    a = mge.tensor(a_val)
    b = mge.tensor(b_val)

    k = 0.1
    c = scaled_add_external(a, b, k)
    print(c) # 输出 Tensor([1.4 2.5 3.6])


更多示例
------------------------------

更一般的例子，请参考 MegEngine 中 `PyTorch 子图的实现 <https://github.com/MegEngine/MegEngine/tree/master/python_module/megengine/module/pytorch>`_ 。
