/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-10 20:16:24
*/
#include "YOLOv562Detector.h"
#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

// test_image "/userdata/yolov5s_v6.2_output3_4.rknn"  80  640
// 1 模型路径 /userdata/yolov5s_v6.2_output3_4.rknn
// 2 模型推理类别个数 80
// 3 模型输入大小 640

int main(int argc, char *argv[])
{

    std::string model_file_path = std::string(argv[1]); // 模型路径
    int number_class = std::stoi(std::string(argv[2]));
    int input_size = std::stoi(std::string(argv[3]));

    ModelConfig cfg;
    cfg.nc = number_class;
    cfg.model_file_path = model_file_path.c_str();
    cfg.nms_threshold = 0.45;
    cfg.conf_thres = 0.25;


    YOLOv562Detector detector;
    detector.load_model(&cfg);

    // test image

    cv::Mat orig_img = cv::imread("1.jpg");

    detect_result_group_t detect_result_group = detector.inference(orig_img, input_size);

    for (int j = 0; j < detect_result_group.count; j++)
    {
        detect_result_t *det_result = &(detect_result_group.results[j]);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        cv::rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);

        cv::putText(orig_img, cv::format("[%d]#[%f]", det_result->class_id, det_result->prop), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, .6, cv::Scalar(0, 255, 0));
    }

    std::string output_file = cv::format("out%d.jpg", 1);
    cv::imwrite(output_file, orig_img);

    return 0;
}
