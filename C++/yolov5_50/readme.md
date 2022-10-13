开发语言：C++  
YOLOv5版本：YOLOv5_50  
设备测试环境：RV1126  
交叉编译环境：Ubuntu18.04  

一共三块

```
YOLOv550Detector.h
YOLOv550Detector.cpp
test_video.cc 主函数
```

生成的test_video程序可以直接拷贝到开发板中进行模型测试

test_video的使用方法

test_video arg1 arg2 arg3  
arg1 是模型所在的路径  
arg2 是模型推理的个数  
arg3 是模型输入大小 640  
默认推理1.mp4 1280×720视频，可根据自己需求更改  
例如  
```
 ./test_video /userdata/model/rv1109_rv1126/coco.rknn 80 640  
 ```
默认推理视频文件时1.mp4 1280×720视频  
推理结果是out.avi文件，可根据自己需求更改。  

YOLOv550Detector类的使用方法  

模型配置

```
ModelConfig cfg;
cfg.nc = number_class;
cfg.model_file_path = model_file_path;
```

声明一个检测器对象，加载配置

```
YOLOv550Detector Detector;
Detector.load_model(&cfg);
```

推理

```
detect_result_group_t detect_result_group = YOLOv550Detector.inference(frame);
```
        
绘制推理结果

```
for (int j = 0; j < detect_result_group.count; j++)
{
    detect_result_t *det_result = &(detect_result_group.results[j]);

    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);

    cv::putText(frame, cv::format("[%d]#[%f]", det_result->class_id, det_result->prop), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, .6, cv::Scalar(0, 255, 0));
}
```

完整的使用方法



```
    std::string model_file_path = std::string(argv[1]); // 模型路径
    int number_class = std::stoi(std::string(argv[2]));
   int input_size = std::stoi(std::string(argv[3]));


    ModelConfig cfg;
    cfg.nc = number_class;
    cfg.model_file_path = model_file_path.c_str();

    YOLOv550Detector detector;
    detector.load_model(&cfg);


    cv::VideoCapture capture("./1.mp4");

    if (!capture.isOpened())
    {
        printf("could not read this video file...\n");
        return -1;
    }

    int num_frames = 0;

    cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(1280, 720));

    while (true)
    {
        cv::Mat frame;

        if (!capture.read(frame)) // if not success, break loop
        {
            std::cout << "\n Cannot read the video file. please check your video.\n";
            break;
        }
        detect_result_group_t detect_result_group = detector.inference(frame,input_size);
        num_frames++;
         std::cout <<num_frames<<std::endl;

        for (int j = 0; j < detect_result_group.count; j++)
        {
            detect_result_t *det_result = &(detect_result_group.results[j]);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);

            cv::putText(frame, cv::format("[%d]#[%f]", det_result->class_id, det_result->prop), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, .6, cv::Scalar(0, 255, 0));
        }

        video.write(frame);
    }
    capture.release();
    video.release();
```
