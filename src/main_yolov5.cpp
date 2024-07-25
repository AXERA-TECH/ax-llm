#include "runner/ax_model_runner/ax_model_runner_ax650.hpp"
#include "runner/ax_model_runner/ax_model_runner_ax650_host.hpp"
#include "cmdline.hpp"
#include "opencv2/opencv.hpp"

typedef struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
} Object;

void get_input_data_letterbox(cv::Mat mat, std::vector<uint8_t> &image, int letterbox_rows, int letterbox_cols, bool bgr2rgb = false)
{
    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / mat.rows) < (letterbox_cols * 1.0 / mat.cols))
    {
        scale_letterbox = (float)letterbox_rows * 1.0f / (float)mat.rows;
    }
    else
    {
        scale_letterbox = (float)letterbox_cols * 1.0f / (float)mat.cols;
    }
    resize_cols = int(scale_letterbox * (float)mat.cols);
    resize_rows = int(scale_letterbox * (float)mat.rows);

    cv::Mat img_new(letterbox_rows, letterbox_cols, CV_8UC3, image.data());

    cv::resize(mat, mat, cv::Size(resize_cols, resize_rows));

    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;

    // Letterbox filling
    cv::copyMakeBorder(mat, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (bgr2rgb)
    {
        cv::cvtColor(img_new, img_new, cv::COLOR_BGR2RGB);
    }
}

template <typename T>
static inline float intersection_area(const T &a, const T &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

template <typename T>
static void qsort_descent_inplace(std::vector<T> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

template <typename T>
static void qsort_descent_inplace(std::vector<T> &faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

template <typename T>
static void nms_sorted_bboxes(const std::vector<T> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const T &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const T &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void get_out_bbox(std::vector<Object> &proposals, std::vector<Object> &objects, const float nms_threshold, int letterbox_rows, int letterbox_cols, int src_rows, int src_cols)
{
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    /* yolov5 draw the result */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / src_rows) < (letterbox_cols * 1.0 / src_cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / src_rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / src_cols;
    }
    resize_cols = int(scale_letterbox * src_cols);
    resize_rows = int(scale_letterbox * src_rows);

    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)src_rows / resize_rows;
    float ratio_y = (float)src_cols / resize_cols;

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
        x1 = (x1 - tmp_w) * ratio_x;
        y1 = (y1 - tmp_h) * ratio_y;

        x0 = std::max(std::min(x0, (float)(src_cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(src_rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(src_cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(src_rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void generate_proposals_yolov5(int stride, const float *feat, float prob_threshold, std::vector<Object> &objects,
                               int letterbox_cols, int letterbox_rows, const float *anchors, float prob_threshold_unsigmoid, int cls_num = 80)
{
    int anchor_num = 3;
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int anchor_group;
    if (stride == 8)
        anchor_group = 1;
    if (stride == 16)
        anchor_group = 2;
    if (stride == 32)
        anchor_group = 3;

    auto feature_ptr = feat;

    for (int h = 0; h <= feat_h - 1; h++)
    {
        for (int w = 0; w <= feat_w - 1; w++)
        {
            for (int a = 0; a <= anchor_num - 1; a++)
            {
                if (feature_ptr[4] < prob_threshold_unsigmoid)
                {
                    feature_ptr += (cls_num + 5);
                    continue;
                }

                // process cls score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s <= cls_num - 1; s++)
                {
                    float score = feature_ptr[s + 5];
                    if (score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }
                // process box score
                float box_score = feature_ptr[4];
                float final_score = sigmoid(box_score) * sigmoid(class_score);

                if (final_score >= prob_threshold)
                {
                    float dx = sigmoid(feature_ptr[0]);
                    float dy = sigmoid(feature_ptr[1]);
                    float dw = sigmoid(feature_ptr[2]);
                    float dh = sigmoid(feature_ptr[3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj);
                }

                feature_ptr += (cls_num + 5);
            }
        }
    }
}

const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;

const char *CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};
const float ANCHORS[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

const int DEFAULT_LOOP_COUNT = 1;

const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.45f;

int main(int argc, char **argv)
{
    cmdline::parser parser;
    parser.add<std::string>("model", 'm', "model file", true);
    parser.add<std::string>("image", 'i', "image file", true);
    parser.parse_check(argc, argv);

    std::string model_file = parser.get<std::string>("model");
    std::string image_file = parser.get<std::string>("image");
    cv::Mat img = cv::imread(image_file);
    if (img.empty())
    {
        std::cout << "load img error" << std::endl;
        return -1;
    }

    std::vector<uint8_t> image(DEFAULT_IMG_H * DEFAULT_IMG_W * 3, 0);
    cv::Mat mat = cv::imread(image_file);
    if (mat.empty())
    {
        fprintf(stderr, "Read image failed.\n");
        return -1;
    }
    get_input_data_letterbox(mat, image, DEFAULT_IMG_H, DEFAULT_IMG_W, true);

    ax_runner_ax650 runner;
    if (runner.init(model_file.c_str()) != 0)
    {
        std::cout << "init model error" << std::endl;
        return -1;
    }
    memcpy(runner.get_input(0).pVirAddr, image.data(), image.size());

    // dump input data
    FILE *fp = fopen("input.bin", "wb");
    fwrite(image.data(), image.size(), 1, fp);
    fclose(fp);

    runner.inference();

    ax_runner_ax650_host runner_host;
    if (runner_host.init(model_file.c_str()) != 0)
    {
        std::cout << "init model error" << std::endl;
        return -1;
    }
    memcpy(runner_host.get_input(0).pVirAddr, image.data(), image.size());
    runner_host.inference();

    std::vector<Object> proposals;
    std::vector<Object> objects;
    float prob_threshold_u_sigmoid = -1.0f * (float)std::log((1.0f / PROB_THRESHOLD) - 1.0f);

    for (uint32_t i = 0; i < 3; ++i)
    {
        auto ptr = (float *)runner.get_output(i).pVirAddr;
        auto ptr_host = (float *)runner_host.get_output(i).pVirAddr;

        char str[256] = {0};
        
        sprintf(str, "output_slave_%d.bin", i);
        FILE *fp = fopen(str, "wb");
        fwrite(ptr, runner.get_output(i).nSize, 1, fp);
        fclose(fp);

        sprintf(str, "output_host_%d.bin", i);
        FILE *fp_host = fopen(str, "wb");
        fwrite(ptr_host, runner_host.get_output(i).nSize, 1, fp_host);
        fclose(fp_host);

        printf("slave:%9.6f %9.6f %9.6f %9.6f %9.6f\n host:%9.6f %9.6f %9.6f %9.6f %9.6f\n\n",
               ptr[0], ptr[1], ptr[2], ptr[3], ptr[4],
               ptr_host[0], ptr_host[1], ptr_host[2], ptr_host[3], ptr_host[4]);

        int32_t stride = (1 << i) * 8;
        generate_proposals_yolov5(stride, ptr, PROB_THRESHOLD, proposals, DEFAULT_IMG_W, DEFAULT_IMG_H, ANCHORS, prob_threshold_u_sigmoid);
    }

    get_out_bbox(proposals, objects, NMS_THRESHOLD, DEFAULT_IMG_H, DEFAULT_IMG_W, mat.rows, mat.cols);

    for (auto obj : objects)
    {
        cv::Rect r = obj.rect;
        cv::rectangle(img, r, cv::Scalar(0, 255, 0), 2, 8);
        cv::putText(img, CLASS_NAMES[obj.label], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
    }

    cv::imwrite("result.jpg", img);

    return 0;
}