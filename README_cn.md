# deeplabv3p_hobot_dnn
## 简介
使用基于ADE20K数据集训练的Deeplabv3p语义分割，分割室内场景。
<video src="demo.mp4"></video>


## 数据集: ADE20K 2016
从150类别中筛选出以下10个类别, 其中, empty是原本为空, others是原本为其他类别
```
"empty", "others", "floor", "bed", "window", "person", "door", "table", "pall", "chair"
```
数据集下载
```bash
wget -O ./data/ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```

## 模型：DeepLabv3plus_mobilenet
仓库地址2：https://github.com/bubbliiiing/deeplabv3-plus-pytorch （转化为VOC格式即可训练）
作者CSDN：https://blog.csdn.net/weixin_44791964/article/details/120113686
LICENSE：MIT License
导出：
导出前修改检测头, 去掉尾部的Resize算子, 直接输出小的特征图, 同时进行transpose.
deeplabv3-plus-pytorch/nets/deeplabv3_plus.py -> class:DeepLab -> method:forward
```python
def forward(self, x):
    H, W = x.size(2), x.size(3)
    low_level_features, x = self.backbone(x)
    x = self.aspp(x)
    low_level_features = self.shortcut_conv(low_level_features)
    x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
    x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
    x = self.cls_conv(x)
    x = x.permute(0, 2, 3, 1)
    # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
    return x
```
export.py
```python
import numpy as np
from PIL import Image

from deeplab import DeeplabV3
import onnx, torch
from onnxsim import simplify

onnx_save_path  = "model_data/models.onnx"

deeplab = DeeplabV3()
deeplab.generate(onnx=True)

# Export the model
print(f'Starting export with onnx {onnx.__version__}.')
im = torch.zeros(1, 3, 640, 640).to('cpu')  # image size(1, 3, 512, 512) BCHW
torch.onnx.export(deeplab.net,
                im,
                f               = onnx_save_path,
                verbose         = False,
                opset_version   = 11,
                training        = torch.onnx.TrainingMode.EVAL,
                do_constant_folding = True,
                input_names     = ["images"],
                output_names    = ["output"],
                dynamic_axes    = None)

# Checks
model_onnx = onnx.load(onnx_save_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

model_onnx, check = simplify(
    model_onnx,
    dynamic_input_shape=False,
    input_shapes=None)
assert check, 'assert check failed'
onnx.save(model_onnx, onnx_save_path)

print('Onnx model save as {}'.format(onnx_save_path))
```

## 推理程序使用方法
使用平台：X5 Ubuntu，需要安装好ROS2 Humble和TROS。
使用方法：
1. 修改相关路径和配置项目

 - deeplabv3p_hobot_dnn/src/parser.cpp 文件中
30, 31行：输出头分辨率160;
45, 47行：类别数量10

 - deeplabv3p_hobot_dnn/src/sample.cpp 文件中
32行：节点名称deeplabv3p_hobot_dnn;
71行：订阅的话题名称;
76行：发布的话题名称;
82行：读取的bin模型路径;

2. 编译
```bash
rm -rf build install log
colcon build --packages-select deeplabv3p_hobot_dnn --parallel-workers 1 --executor sequential
```

3. 运行

根据需要source
```bash
source /opt/ros/humble/setup.bash
source /opt/tros/humble/setup.bash
# source install/setup.bash
```

启动USB摄像头节点，发布MJPEG图像
```bash
ros2 run hobot_usb_cam hobot_usb_cam \
--ros-args --log-level warn \
--ros-args -p zero_copy:=False \
--ros-args -p io_method:=mmap \
--ros-args -p video_device:=/dev/video0 \
--ros-args -p pixel_format:=mjpeg \
--ros-args -p image_height:=480 \
--ros-args -p image_width:=640
```

启动codec节点，MJPEG转码为nv12
```bash
ros2 run hobot_codec hobot_codec_republish \
--ros-args --log-level warn \
--ros-args -p channel:=1 \
--ros-args -p in_format:=jpeg \
--ros-args -p sub_topic:=image \
--ros-args -p in_mode:=ros \
--ros-args -p out_format:=nv12 \
--ros-args -p out_mode:=shared_mem \
--ros-args -p pub_topic:=hbmem_img
```

启动算法推理节点
```bash
source install/setup.bash
ros2 run deeplabv3p_hobot_dnn deeplabv3p_hobot_dnn
```