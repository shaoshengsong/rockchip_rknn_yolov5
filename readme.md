## 模型导出环境  
rknn-toolkit 1.7.1

## 部署环境
设备测试环境：RV1126  
交叉编译环境：Ubuntu18.04  

当一个yolov5_6.2模型训练完成后

## 模型导出
### 1 pt模型 转 onnx
进入rockchip_rknn_yolov5\yolov5_6.2_export目录  
执行 
``` 
python export.py --weights yolov5s_v6.2.pt --img 640 --batch 1 --include onnx torchscript  
```
yolov5s_v6.2.pt 是自己训练的模型  
会生成onnx模型  

###  2 onnx模型 转 rknn


进入rockchip_rknn_yolov5\yolov5_6.2_export目录  
执行  
```
python yolov562_to_rknn_3_4.py
```

## C++ 部署
YOLOv562Detector.cpp  
YOLOv562Detector.h  
test_image.cc

文件所在路径rockchip_rknn_yolov5\C++\yolov5_62

使用方法  
```
test_image "/userdata/yolov5s_v6.2_output3_4.rknn"  80  640
```
1 模型路径 /userdata/yolov5s_v6.2_output3_4.rknn  
2 模型推理类别个数 80   
3 模型输入大小 640

