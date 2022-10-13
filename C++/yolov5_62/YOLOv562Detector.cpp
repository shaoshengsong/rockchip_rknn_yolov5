/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-10 20:16:03
*/
#include "YOLOv562Detector.h"

YOLOv550Detector::~YOLOv550Detector()
{
    int ret = rknn_destroy(ctx);
    printf("destroy\n");
}

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

unsigned char *YOLOv550Detector::load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

unsigned char *YOLOv550Detector::load_file(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int YOLOv550Detector::load_model(ModelConfig *cfg)
{

    nc_ = cfg->nc;
    nms_threshold_ = cfg->nms_threshold;
    conf_thres_ = cfg->conf_thres;
    PROP_BOX_SIZE_ = 5 + nc_;

    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_file(cfg->model_file_path, &model_data_size);

    int ret = 0;
    ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num_.n_input, io_num_.n_output);

    rknn_tensor_attr input_attrs[io_num_.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num_.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
    }

    output_attrs_ = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num_.n_output);
    memset(output_attrs_, 0, sizeof(rknn_tensor_attr) * io_num_.n_output);

    for (uint32_t i = 0; i < io_num_.n_output; i++)
    {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                         sizeof(rknn_tensor_attr));
    }

    channel_ = 3;

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        width_ = input_attrs[0].dims[0];
        height_ = input_attrs[0].dims[1];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        width_ = input_attrs[0].dims[1];
        height_ = input_attrs[0].dims[2];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height_, width_, channel_);
    return 0;
}
detect_result_group_t YOLOv550Detector::inference(cv::Mat &orig_img, int scale_size)
{

    int img_width = orig_img.cols;
    int img_height = orig_img.rows;

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width_ * height_ * channel_;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    cv::Mat resimg;
    cv::resize(orig_img, resimg, cv::Size(scale_size, scale_size), (0, 0), (0, 0), cv::INTER_LINEAR);
    cv::cvtColor(resimg, resimg, cv::COLOR_BGR2RGB);

    inputs[0].buf = resimg.data;

    rknn_inputs_set(ctx, io_num_.n_input, inputs);

    outputs_ = (rknn_output *)malloc(sizeof(rknn_output) * io_num_.n_output);
    memset(outputs_, 0, sizeof(rknn_output) * io_num_.n_output);

    for (int i = 0; i < io_num_.n_output; i++)
    {
        outputs_[i].want_float = 0;
    }

    int ret = 0;
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num_.n_output, outputs_, NULL);

    // post process
    float scale_w = (float)width_ / img_width;
    float scale_h = (float)height_ / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;

    std::vector<uint32_t> out_zps;
    for (uint32_t i = 0; i < io_num_.n_output; ++i)
    {
        out_scales.push_back(output_attrs_[i].scale);
        out_zps.push_back(output_attrs_[i].zp);
    }

    post_process((uint8_t *)outputs_[0].buf, (uint8_t *)outputs_[1].buf, (uint8_t *)outputs_[2].buf, height_, width_,
                 conf_thres_, nms_threshold_, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    ret = rknn_outputs_release(ctx, io_num_.n_output, outputs_);

    return detect_result_group;
}

float YOLOv550Detector::CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int YOLOv550Detector::nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order, int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1 || classIds[i] != filterId)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

int YOLOv550Detector::quick_sort_indice_inverse(
    std::vector<float> &input,
    int left,
    int right,
    std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

float YOLOv550Detector::sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

float YOLOv550Detector::unsigmoid(float y)
{
    return -1.0 * logf((1.0 / y) - 1.0);
}

int32_t YOLOv550Detector::__clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

uint8_t YOLOv550Detector::qnt_f32_to_affine(float f32, uint32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

float YOLOv550Detector::deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

int YOLOv550Detector::process(uint8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                              std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                              float threshold, uint32_t zp, float scale)
{

    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid(threshold);
    uint8_t thres_u8 = qnt_f32_to_affine(thres, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                uint8_t box_confidence = input[(PROP_BOX_SIZE_ * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_u8)
                {
                    int offset = (PROP_BOX_SIZE_ * a) * grid_len + i * grid_w + j;
                    uint8_t *in_ptr = input + offset;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);

                    uint8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < nc_; ++k)
                    {
                        uint8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }

                    float deqnt_cls_conf = sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale));

                    if (deqnt_cls_conf > conf_thres_)
                    {
                        float deqnt_box_conf = sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale));
                        objProbs.push_back(deqnt_box_conf * deqnt_cls_conf);

                        classId.push_back(maxClassId);

                        validCount++;
                    }
                }
            }
        }
    }
    return validCount;
}

int YOLOv550Detector::clamp(float val, int min, int max)
{
    return val > min ? (val < max ? val : max) : min;
}

int YOLOv550Detector::post_process(uint8_t *input0, uint8_t *input1, uint8_t *input2, int model_in_h, int model_in_w,
                                   float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                                   std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales,
                                   detect_result_group_t *group)
{

    memset(group, 0, sizeof(detect_result_group_t));

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w,
                          stride0, filterBoxes, objProbs, classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w,
                          stride1, filterBoxes, objProbs, classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w,
                          stride2, filterBoxes, objProbs, classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;
    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {

        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        int left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        int top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        int right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        int bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);

        if ((right - left) <= 2)
            continue;

        if ((bottom - top) <= 2)
            continue;
        //------------------------------------------------------------------------------------------------------------------------
        group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);

        group->results[last_count].prop = obj_conf;
        group->results[last_count].class_id = id;

        last_count++;
    }
    group->count = last_count;

    return 0;
}

void YOLOv550Detector::create_rknn_list(rknn_list **s)
{
    if (*s != NULL)
        return;
    *s = (rknn_list *)malloc(sizeof(rknn_list));
    (*s)->top = NULL;
    (*s)->size = 0;
    printf("create rknn_list success\n");
}

void YOLOv550Detector::destory_rknn_list(rknn_list **s)
{
    Node *t = NULL;
    if (*s == NULL)
        return;
    while ((*s)->top)
    {
        t = (*s)->top;
        (*s)->top = t->next;
        free(t);
    }
    free(*s);
    *s = NULL;
}

void YOLOv550Detector::rknn_list_push(rknn_list *s, long timeval,
                                      detect_result_group_t detect_result_group)
{
    Node *t = NULL;
    t = (Node *)malloc(sizeof(Node));
    t->timeval = timeval;
    t->detect_result_group = detect_result_group;
    if (s->top == NULL)
    {
        s->top = t;
        t->next = NULL;
    }
    else
    {
        t->next = s->top;
        s->top = t;
    }
    s->size++;
}

void YOLOv550Detector::rknn_list_pop(rknn_list *s, long *timeval,
                                     detect_result_group_t *detect_result_group)
{
    Node *t = NULL;
    if (s == NULL || s->top == NULL)
        return;
    t = s->top;
    *timeval = t->timeval;
    *detect_result_group = t->detect_result_group;
    s->top = t->next;
    free(t);
    s->size--;
}

void YOLOv550Detector::rknn_list_drop(rknn_list *s)
{
    Node *t = NULL;
    if (s == NULL || s->top == NULL)
        return;
    t = s->top;
    s->top = t->next;
    free(t);
    s->size--;
}

int YOLOv550Detector::rknn_list_size(rknn_list *s)
{
    if (s == NULL)
        return -1;
    return s->size;
}