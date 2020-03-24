#include <stdlib.h>
#include <iostream>
#include "megbrain/serialization/serializer.h"
using namespace mgb;

cg::ComputingGraph::OutputSpecItem make_callback_copy(SymbolVar dev,
                                                      HostTensorND& host) {
    auto cb = [&host](DeviceTensorND& d) { host.copy_from(d); };
    return {dev, cb};
}

int main(int argc, char* argv[]) {
    // 运行编译后的该程序，需要提供模型文件名、用于进行异或操作的两个值（x 和 y）
    std::cout << " Usage: ./xornet_deploy model_name x_value y_value"
              << std::endl;
    if (argc != 4) {
        std::cout << " Wrong argument" << std::endl;
        return 0;
    }
    // 读取通过运行参数指定的模型文件
    std::unique_ptr<serialization::InputFile> inp_file =
            serialization::InputFile::make_fs(argv[1]);
    // 加载通过运行参数指定的计算输入
    float x = atof(argv[2]);
    float y = atof(argv[3]);
    // 使用 GraphLoader 将模型文件转成 LoadResult，包括了计算图和输入等信息
    auto loader = serialization::GraphLoader::make(std::move(inp_file));
    serialization::GraphLoadConfig config;
    serialization::GraphLoader::LoadResult network =
            loader->load(config, false);
    // 通过 dump 时指定的名称拿到输入 Tensor
    auto data = network.tensor_map["data"];
    // 给输入 Tensor 赋值
    float* data_ptr = data->resize({1, 2}).ptr<float>();
    data_ptr[0] = x;
    data_ptr[1] = y;

    // 将网络编译为异步执行函数
    // 输出output_var为一个字典的列表，second拿到键值对中的值，并存在 predict 中
    HostTensorND predict;
    std::unique_ptr<cg::AsyncExecutable> func =
            network.graph->compile({make_callback_copy(
                    network.output_var_map.begin()->second, predict)});
    func->execute();
    func->wait();
    // 输出值为对输入计算异或值 0 和 1 两个类别的概率
    float* predict_ptr = predict.ptr<float>();
    std::cout << " Predicted: " << predict_ptr[0] << " " << predict_ptr[1]
              << std::endl;
}
