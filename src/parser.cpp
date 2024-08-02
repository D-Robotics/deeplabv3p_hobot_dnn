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

#include "include/parser.h"

#include <memory>

using hobot::dnn_node::DNNTensor;

namespace hobot {
namespace dnn_node {
namespace dnn_node_sample {

int32_t ParseDeepLabv3p(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<sensor_msgs::msg::Image> &image_msg) {
    // 创建发布的消息, 设置图像的宽度、高度和步长
    // auto image_msg = std::make_shared<sensor_msgs::msg::Image>(); 
    image_msg->width = 160;
    image_msg->height = 160;
    image_msg->step = image_msg->width;

    // 创建图像数据
    std::vector<uint8_t> image_data(image_msg->width * image_msg->height);
    
    // 找到最大值
    hbSysFlushMem(&(node_output->output_tensors[0]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    // 基准地址
    float* data = reinterpret_cast<float*>(node_output->output_tensors[0]->sysMem[0].virAddr);
    for(int col=0; col<160; col++){
        for(int row=0; row<160; row++){
            // 偏移到现在C通道的地址上
            int cnt_one = 160*col + row; // 一级偏移量
            float* c_data = data + cnt_one * 10; // 10是channel
            uint8_t max_index = 0;
            for(uint8_t i=1; i<10; i++){
                if(c_data[i] > c_data[max_index])
                    max_index = i;
            }
            image_data[cnt_one] = max_index;
        }
    }
    // 将图像数据复制到消息中
    image_msg->data = image_data;

  return 0;
}

}  // namespace dnn_node_sample
}  // namespace dnn_node
}  // namespace hobot