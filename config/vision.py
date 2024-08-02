# 需要在同局域网的Ubuntu机器运行
# 请勿在板端运行
# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy         
from rclpy.node import Node            
from sensor_msgs.msg import CompressedImage, Image

from time import time, sleep
import numpy as np
import cv2, os

class VISION(Node):
    def __init__(self, name):
        super().__init__(name)
        # ros2
        self.jpeg_sub = self.create_subscription(CompressedImage, 'image', self.jpeg_sub_callback, 10)
        self.mask_sub = self.create_subscription(Image, 'deeplabv3p_mask', self.mask_sub_callback, 10)
        

    def jpeg_sub_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print(image_np.shape)
        
        # 显示图像
        #cv2.imshow('Image', image_np)
        #cv2.waitKey(1)
    def mask_sub_callback(self, msg):
         np_arr = np.frombuffer(msg.data, np.uint8).reshape(160,160)
         print(np_arr.shape)

def main(args=None):                                        # ROS2节点主入口main函数
    rclpy.init(args=args)                                   # ROS2 Python接口初始化
    node = VISION("vision")                # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                                        # 循环等待ROS2退出
    node.destroy_node()                                     # 销毁节点对象
    rclpy.shutdown() 
    
if __name__ == "__main__":
    main()
