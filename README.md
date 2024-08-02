English| [简体中文](./README_cn.md)
# deeplabv3p_hobot_dnn
## Introduction
The Deeplabv3p semantic segmentation trained on the ADE20K dataset was used to segment indoor scenes.
<video src="demo.mp4"></video>
​

## Dataset: ADE20K 2016
From the 150 categories, the following 10 categories are selected, where empty is meant to be empty and others is meant to be other categories
```
"empty", "others", "floor", "bed", "window", "person", "door", "table", "pall", "chair"
```
Dataset download
```bash
wget -O ./data/ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```
​
## Model: DeepLabv3plus_mobilenet
Warehouse address 2:https://github.com/bubbliiiing/deeplabv3-plus-pytorch (into VOC format can training)
The author CSDN:https://blog.csdn.net/weixin_44791964/article/details/120113686
LICENSE: MIT License
Export:
Before export, modify the detection head, remove the Resize operator in the tail, and directly output the small feature map, while transpose.
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
​
from deeplab import DeeplabV3
import onnx, torch
from onnxsim import simplify
​
onnx_save_path  = "model_data/models.onnx"
​
deeplab = DeeplabV3()
deeplab.generate(onnx=True)
​
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
​
# Checks
model_onnx = onnx.load(onnx_save_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model
​
model_onnx, check = simplify(
model_onnx,
dynamic_input_shape=False,
input_shapes=None)
assert check, 'assert check failed'
onnx.save(model_onnx, onnx_save_path)
​
print('Onnx model save as {}'.format(onnx_save_path))
```
​
## How the reasoner works
Platform: X5 Ubuntu with ROS2 Humble and TROS installed.
Usage:
1. Change the relevant paths and configure the project
​
-deeplabv3p_hobot_dnn /src/parser.cpp
Lines 30, 31: output head resolution 160;
Line 45, 47: Number of categories 10
​
-deeplabv3p_hobot_dnn /src/sample.cpp file
Line 32: node name deeplabv3p_hobot_dnn;
Line 71: topic name to subscribe to;
Line 76: published topic name;
Line 82: bin model path to read;
​
Compilation
```bash
rm -rf build install log
colcon build --packages-select deeplabv3p_hobot_dnn --parallel-workers 1 --executor sequential
```
​
Step 3 Run
​
source as needed
```bash
source /opt/ros/humble/setup.bash
source /opt/tros/humble/setup.bash
# source install/setup.bash
```
​
Start the USB camera node and publish the MJPEG image
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
​
The codec node is started and MJPEG is transcoded to nv12
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

run AI node
```bash
source install/setup.bash
ros2 run deeplabv3p_hobot_dnn deeplabv3p_hobot_dnn
```