// Copyright (c) 2022，Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

# include "dnn_node/dnn_node_data.h"
# include "sensor_msgs/msg/image.hpp"  // DeepLabv3p的算法输出结果存储的变量类型的头文件

namespace hobot {
namespace dnn_node {
namespace dnn_node_sample {
// 定义算法输出数据类型
// sensor_msgs::msg::Image

// 自定义的算法输出解析方法
// - 参数
//   - [in] node_output dnn node输出，包含算法推理输出
//          解析时，如果不需要使用前处理参数，可以直接使用DnnNodeOutput中的
//          std::vector<std::shared_ptr<DNNTensor>>
//          output_tensors成员作为Parse的入口参数
//   - [in/out] results 解析后的结构化数据, deeplabv3p的后处理结果为160*160的灰度图，对应像素的位置存储类别index
// - 返回值
//   - 0 成功
//   - -1 失败
int32_t ParseDeepLabv3p(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<sensor_msgs::msg::Image> &results);


}  // namespace dnn_node_sample
}  // namespace dnn_node
}  // namespace hobot