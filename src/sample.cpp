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

#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "hobot_cv/hobotcv_imgproc.h"
#include "include/parser.h"
#include "sensor_msgs/msg/image.hpp"


struct DNNNodeSampleOutput : public hobot::dnn_node::DnnNodeOutput {
  // 缩放比例系数，原图和模型输入分辨率的比例。
  float ratio = 1.0;
};

// 继承DnnNode虚基类，创建算法推理节点
class DNNNodeSample : public hobot::dnn_node::DnnNode {
 public:
  DNNNodeSample(const std::string& node_name = "deeplabv3p_hobot_dnn",
                const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 protected:
  // 实现基类的纯虚接口，用于配置Node参数
  int SetNodePara() override;
  // 实现基类的虚接口，将解析后结构化的算法输出数据封装成ROS Msg后发布
  int PostProcess(const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>&
                      node_output) override;

 private:
  // 算法输入图片数据的宽和高
  int model_input_width_ = -1;
  int model_input_height_ = -1;

  // 图片消息订阅者
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::ConstSharedPtr
      ros_img_subscription_ = nullptr;
  // 算法推理结果消息发布者
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr msg_publisher_ =
      nullptr;

  // 图片消息订阅回调
  void FeedImg(const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg);
};

DNNNodeSample::DNNNodeSample(const std::string& node_name,
                             const rclcpp::NodeOptions& options)
    : hobot::dnn_node::DnnNode(node_name, options) {
  // Init中使用DNNNodeSample子类实现的SetNodePara()方法进行算法推理的初始化
  if (Init() != 0 ||
      GetModelInputSize(0, model_input_width_, model_input_height_) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn_node_sample"), "Node init fail!");
    rclcpp::shutdown();
  }

  // 创建消息订阅者，从摄像头节点订阅图像消息
  ros_img_subscription_ =
      this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
          "/hbmem_img",
          rclcpp::SensorDataQoS(),
          std::bind(&DNNNodeSample::FeedImg, this, std::placeholders::_1));
  // 创建消息发布者，发布算法推理消息
  msg_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/deeplabv3p_mask", 10);
}

int DNNNodeSample::SetNodePara() {
  if (!dnn_node_para_ptr_) return -1;
  // 指定算法推理使用的模型文件路径
  dnn_node_para_ptr_->model_file = "/userdata/hobot_dnn/deeplabv3p_ade20k_nc10_best_Bayese_nv12.bin";
  // 指定算法推理任务类型
  // 本示例使用的人体检测算法输入为单张图片，对应的算法推理任务类型为ModelInferType
  // 只有当算法输入为图片和roi（Region of
  // Interest，例如目标的检测框）时，算法推理任务类型为ModelRoiInferType
  dnn_node_para_ptr_->model_task_type =
      hobot::dnn_node::ModelTaskType::ModelInferType;
  // 指定算法推理使用的任务数量
  dnn_node_para_ptr_->task_num = 2;
  // 不通过bpu_core_ids参数指定算法推理使用的BPU核，使用负载均衡模式

  return 0;
}

void DNNNodeSample::FeedImg(
    const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr img_msg) {
  if (!rclcpp::ok() || !img_msg) {
    return;
  }

  // 1 对订阅到的图片消息进行验证，本示例只支持处理NV12格式图片数据
  // 如果是其他格式图片，订阅hobot_codec解码/转码后的图片消息
  if ("nv12" !=
      std::string(reinterpret_cast<const char*>(img_msg->encoding.data()))) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn_node_sample"),
                 "Only support nv12 img encoding! Using hobot codec to process "
                 "%d encoding img.",
                 img_msg->encoding.data());
    return;
  }

  // 2 创建算法输出数据，填充消息头信息，用于推理完成后AI结果的发布
  auto dnn_output = std::make_shared<DNNNodeSampleOutput>();
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->msg_header->set__stamp(img_msg->time_stamp);

  // 3 算法前处理，即创建算法输入数据
  std::shared_ptr<hobot::dnn_node::NV12PyramidInput> pyramid = nullptr;
  cv::Mat src(img_msg->height * 3 / 2, img_msg->width, CV_8UC1, (void*)(reinterpret_cast<const char*>(img_msg->data.data())));
  cv::Mat out_img;
  if (hobot_cv::hobotcv_resize(src, img_msg->height, img_msg->width, out_img, model_input_height_, model_input_width_) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn_node_sample"),"hobot_cv::hobotcv_resize resize nv12 img fail!");
    return;
    }
  hobot_cv::hobotcv_resize(src, img_msg->height, img_msg->width, out_img, model_input_height_, model_input_width_);

    uint32_t out_img_width = out_img.cols;
    uint32_t out_img_height = out_img.rows * 2 / 3;
    // 3.2 根据算法输入图片分辨率，使用hobot_dnn中提供的方法创建算法输入数据
    pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
      reinterpret_cast<const char*>(out_img.data),
      out_img_height,
      out_img_width,
      model_input_height_,
      model_input_width_);

  // 3.4 校验算法输入数据
  if (!pyramid) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn_node_sample"), "Get pym fail");
    return;
  }
  // 3.5 将算法输入数据转成dnn node推理输入的格式
  auto inputs =
      std::vector<std::shared_ptr<hobot::dnn_node::DNNInput>>{pyramid};

  // 4
  // 使用创建的算法输入和输出数据，以异步模式运行推理，推理结果通过PostProcess接口回调返回
  if (Run(inputs, dnn_output, nullptr, false) < 0) {
    RCLCPP_INFO(rclcpp::get_logger("dnn_node_sample"), "Run predict fail!");
  }
}

// 推理结果回调，解析算法输出，通过ROS Msg发布消息
int DNNNodeSample::PostProcess(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>& node_output) {
  if (!rclcpp::ok()) {
    return 0;
  }

  // 后处理开始时间
  auto tp_start = std::chrono::system_clock::now();

  // 1 创建用于发布推理结果的ROS Msg
  // ai_msgs::msg::PerceptionTargets::UniquePtr pub_data(
  //     new ai_msgs::msg::PerceptionTargets());
  auto pub_data = std::make_shared<sensor_msgs::msg::Image>();

  // 2 将推理输出对应图片的消息头填充到ROS Msg
  pub_data->set__header(*node_output->msg_header);

  // 3 使用自定义的Parse解析方法，解析算法输出的DNNTensor类型数据
  // 3.1
  // 创建解析输出数据，输出YoloV5Result是自定义的算法输出数据类型，results的维度等于检测出来的目标数
  // std::vector<std::shared_ptr<hobot::dnn_node::dnn_node_sample::YoloV5Result>>
  //     results;

  // 3.2 开始解析
  if (hobot::dnn_node::dnn_node_sample::ParseDeepLabv3p(node_output, pub_data) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("dnn_node_sample"),
                 "Parse node_output fail!");
    return -1;
  }

  // 5 将算法推理输出帧率填充到ROS Msg
  if (node_output->rt_stat) {
    // pub_data->set__fps(round(node_output->rt_stat->output_fps));
    // 如果算法推理统计有更新，输出算法输入和输出的帧率统计、推理耗时
    if (node_output->rt_stat->fps_updated) {
      // 后处理结束时间
      auto tp_now = std::chrono::system_clock::now();
      auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                          tp_now - tp_start)
                          .count();
      RCLCPP_WARN(rclcpp::get_logger("dnn_node_sample"),
                  "input fps: %.2f, out fps: %.2f, infer time ms: %d, "
                  "post process time ms: %d",
                  node_output->rt_stat->input_fps,
                  node_output->rt_stat->output_fps,
                  node_output->rt_stat->infer_time_ms,
                  interval);
    }
  }

  // 6 发布ROS Msg
  msg_publisher_->publish(*pub_data);

  return 0;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DNNNodeSample>());
  rclcpp::shutdown();
  return 0;
}
