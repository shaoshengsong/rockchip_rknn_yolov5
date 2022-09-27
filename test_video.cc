/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-10 20:16:24
*/
#include "YOLOv550Detector.h"
#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>


//使用方法  test_video "/userdata/a.rknn"  9  640 
// test_video 
// 1 模型路径 
// 2 模型推理类别个数 
// 3 模型输入大小 640
//默认推理1.mp4 1280×720视频，可根据自己需求更改
int main(int argc, char *argv[])
{

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

    return 0;
}
