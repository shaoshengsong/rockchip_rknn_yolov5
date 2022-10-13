/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-10 20:15:23
*/
#ifndef YOLOV5_H
#define YOLOV5_H

#include <set>
#include "rknn/rknn_api.h"
#include <opencv2/opencv.hpp>

#define MAXFFRTSPChn 16
#define MODEL_INPUT_SIZE 640

#define MAX_SESSION_NUM MAXFFRTSPChn
#define DRAW_INDEX 0
#define RK_NN_INDEX 0
#define MAX_RKNN_LIST_NUM 10
#define UPALIGNTO(value, align) ((value + align - 1) & (~(align - 1)))
#define UPALIGNTO16(value) UPALIGNTO(value, 16)

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64

typedef struct _ModelConfig
{
    // python
    //  parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    //  parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    int nc = 80;
    const char *model_file_path;
    float nms_threshold = 0.45;
    float conf_thres = 0.25;

} ModelConfig;

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
    int topclass;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE]; // There's no need.
    BOX_RECT box;
    float prop;
    int class_id;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

// rknn list to draw boxs asynchronously
typedef struct node
{
    long timeval;
    detect_result_group_t detect_result_group;
    struct node *next;
} Node;

typedef struct my_stack
{
    int size;
    Node *top;
} rknn_list;

class YOLOv562Detector
{

public:
    //对外接口
    int load_model(ModelConfig *cfg);                                   //加载模型
    detect_result_group_t inference(cv::Mat &in, int scale_size = 640); //推理

    //自动销毁rknn
public:
    int number_;
    rknn_context ctx;
    rknn_input_output_num io_num_;
    YOLOv562Detector() = default;
    ~YOLOv562Detector();
    YOLOv562Detector(const YOLOv562Detector &rhs) = delete;
    YOLOv562Detector &operator=(const YOLOv562Detector &rhs) = delete;

    unsigned char *load_file(const char *filename, int *model_size);
    unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);

private:
    // rknn query 返回的结果，传递给rknn inputs set使用
    int width_;
    int height_;
    int channel_;

    rknn_tensor_attr *output_attrs_;
    rknn_output *outputs_;

private:
    // model
    int nc_;
    float nms_threshold_;
    float conf_thres_;

private:
    int PROP_BOX_SIZE_;

public:
    int clamp(float val, int min, int max);
    float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1);
    int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order, int filterId, float threshold);
    int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);
    float sigmoid(float x);
    float unsigmoid(float y);

    int32_t __clip(float val, float min, float max);
    uint8_t qnt_f32_to_affine(float f32, uint32_t zp, float scale);
    float deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale);
    int process(uint8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                float threshold, uint32_t zp, float scale);

    int post_process(uint8_t *input0, uint8_t *input1, uint8_t *input2, int model_in_h, int model_in_w,
                     float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                     std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales,
                     detect_result_group_t *group);

    rknn_list *rknn_list_[MAXFFRTSPChn];

    void rknn_list_pop(rknn_list *s, long *timeval, detect_result_group_t *detect_result_group);
    void rknn_list_drop(rknn_list *s);
    int rknn_list_size(rknn_list *s);
    void create_rknn_list(rknn_list **s);
    void destory_rknn_list(rknn_list **s);
    void rknn_list_push(rknn_list *s, long timeval, detect_result_group_t detect_result_group);
};

#endif // YOLOV5_H
