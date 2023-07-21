/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
// #include <ncurses.h>

#define _BASETSD_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#undef cimg_display
#define cimg_display 0
#undef cimg_use_jpeg
#define cimg_use_jpeg 1
#undef cimg_use_png
#define cimg_use_png 1
#include "CImg/CImg.h"

#include "drm_func.h"
#include "rga_func.h"
#include "rknn_api.h"
#include "yolo.h"
#include "common.h"
#include "tracker.h"

// #define PLATFORM_RK3588
#define PERF_WITH_POST 0
#define COCO_IMG_NUMBER 5000
#define DUMP_INPUT 0

using namespace cimg_library;
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
           attr->dims[2], attr->dims[3], attr->n_elems, attr->size, 0, attr->type,
           attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
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

static unsigned char *load_model(const char *filename, int *model_size)
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

// static int saveFloat(const char *file_name, float *output, int element_size)
// {
//     FILE *fp;
//     fp = fopen(file_name, "w");
//     for (int i = 0; i < element_size; i++)
//     {
//         fprintf(fp, "%.6f\n", output[i]);
//     }
//     fclose(fp);
//     return 0;
// }

int query_model_info(MODEL_INFO *m, rknn_context ctx)
{
    int ret;
    /* Query sdk version */
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    /* Get input,output attr */
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);
    m->in_nodes = io_num.n_input;
    m->out_nodes = io_num.n_output;
    m->in_attr = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
    m->out_attr = (rknn_tensor_attr *)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
    if (m->in_attr == NULL || m->out_attr == NULL)
    {
        printf("alloc memery failed\n");
        return -1;
    }

    for (int i = 0; i < io_num.n_input; i++)
    {
        m->in_attr[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &m->in_attr[i],
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&m->in_attr[i]);
    }

    for (int i = 0; i < io_num.n_output; i++)
    {
        m->out_attr[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(m->out_attr[i]),
                         sizeof(rknn_tensor_attr));
        printRKNNTensor(&(m->out_attr[i]));
    }

    /* get input shape */
    if (io_num.n_input > 1)
    {
        printf("expect model have 1 input, but got %d\n", io_num.n_input);
        return -1;
    }

    if (m->in_attr[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        m->width = m->in_attr[0].dims[0];
        m->height = m->in_attr[0].dims[1];
        m->channel = m->in_attr[0].dims[2];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        m->width = m->in_attr[0].dims[2];
        m->height = m->in_attr[0].dims[1];
        m->channel = m->in_attr[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", m->height, m->width,
           m->channel);

    return 0;
}

static int status = 0;
static rknn_context ctx;
static rga_context rga_ctx;
static drm_context drm_ctx;
int drm_fd = -1;
int buf_fd = -1; // converted from buffer handle
void *drm_buf = NULL;
unsigned int handle;
MODEL_INFO m_info;
LETTER_BOX letter_box;
static size_t actual_size = 0;
static int img_width = 0;
static int img_height = 0;
static int img_channel = 0;
void *resize_buf;
unsigned char *model_data;

// static const float nms_threshold = NMS_THRESH;
// static const float box_conf_threshold = BOX_THRESH;
static struct timeval start_time, stop_time;
static int ret;
// static rknn_input_output_num io_num;
// static rknn_tensor_attr output_attrs[3];

int DetectorInit(MODEL_INFO *m)
{
    int status = 0;

    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));
    // drm_fd = drm_init(&drm_ctx);

    m->m_type = YOLOX;
    m->color_expect = RK_FORMAT_RGB_888;
    m->anchor_per_branch = 1;
    m->m_path = "../yoloxs_tk2_RK3588_i8.rknn";
    char *anchor_path = " ";
    // 输入图像地址
    m->in_path = "";
    for (int i = 0; i < 18; i++)
    {
        m->anchors[i] = 1;
    }
    if (ret < 0)
        return -1;

    /* Create the neural network */
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(m_info.m_path, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    printf("query info\n");
    ret = query_model_info(&m_info, ctx);
    if (ret < 0)
    {
        return -1;
    }
}

int DetectorRun(cv::Mat &img, std::vector<StObject> &st_objs)
{
    /* Init input tensor */
    rknn_input inputs[1];

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8; /* SAME AS INPUT IMAGE */
    inputs[0].size = m_info.width * m_info.height * m_info.channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC; /* SAME AS INPUT IMAGE */
    inputs[0].pass_through = 0;
    // std::cout << "m_info:whc" << m_info.width << " " << m_info.height << " " << m_info.channel;

    /* Init output tensor */
    rknn_output outputs[m_info.out_nodes];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < m_info.out_nodes; i++)
    {
        // printf("The info type: %d\n", m_info.post_type);
        outputs[i].want_float = m_info.post_type;
    }
    void *resize_buf = malloc(inputs[0].size);
    if (resize_buf == NULL)
    {
        printf("resize buf alloc failed\n");
        return -1;
    }
    void *rk_outputs_buf[m_info.out_nodes];

    img_width = img.cols;
    img_height = img.rows;
    img_channel = img.channels();
    // std::cout << "raw_img:whc" << img_width << " " << img_height << " " << img_channel;
    // drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, img_width, img_height, img_channel * 8,
    //                         &buf_fd, &handle, &actual_size);
    // memcpy(drm_buf, img.data, img_width * img_height * img_channel);
    // memset(resize_buf, 0, inputs[0].size);

    // Letter box resize
    letter_box.target_height = m_info.height;
    letter_box.target_width = m_info.width;
    letter_box.in_height = img_height;
    letter_box.in_width = img_width;
    compute_letter_box(&letter_box);

    // Init rga context
    RGA_init(&rga_ctx);
    // img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, letter_box.resize_width, letter_box.resize_height,
    //                 letter_box.w_pad, letter_box.h_pad, m_info.color_expect,
    //                 letter_box.add_extra_sz_w_pad, letter_box.add_extra_sz_h_pad);
    inputs[0].buf = resize_buf;

    // gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, m_info.in_nodes, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, m_info.out_nodes, outputs, NULL);

    /* Post process */
    detect_result_group_t detect_result_group;
    for (auto i = 0; i < m_info.out_nodes; i++)
        rk_outputs_buf[i] = outputs[i].buf;
    post_process(rk_outputs_buf, &m_info, &letter_box, &detect_result_group);

    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n",
           (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // Draw Objects
    const unsigned char blue[] = {0, 0, 255};
    char score_result[64];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        StObject st_obj;
        detect_result_t *det_result = &(detect_result_group.results[i]);
        printf("%s @ (%d %d %d %d) %f\n",
               det_result->name,
               det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        int ret = snprintf(score_result, sizeof score_result, "%f", det_result->prop);
        // draw box
        if (det_result->prop > 0.5)
        {
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            // cv::putText(img, det_result->name, cv::Point(x1, y1 - 24), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
            // cv::putText(img, score_result, cv::Point(x1, y1 - 12), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
            st_obj.x = detect_result_group.results[i].box.left;
            st_obj.y = detect_result_group.results[i].box.top;
            st_obj.w = detect_result_group.results[i].box.right - st_obj.x;
            st_obj.h = detect_result_group.results[i].box.bottom - st_obj.y;
            st_obj.clsId = detect_result_group.results[i].class_index;
            std::cout << st_obj.x << st_obj.y << st_obj.w << st_obj.h << st_obj.clsId << std::endl;
            st_objs.push_back(st_obj);
        }
    }

    // img.save("./out.bmp");
    // cv::imwrite("./out.jpg",img);
    ret = rknn_outputs_release(ctx, m_info.out_nodes, outputs);
    // drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
}

void DetectorRelease()
{
    // release
    ret = rknn_destroy(ctx);

    RGA_deinit(&rga_ctx);
    if (model_data)
    {
        free(model_data);
    }

    if (m_info.in_attr)
    {
        free(m_info.in_attr);
    }

    if (m_info.out_attr)
    {
        free(m_info.out_attr);
    }
    // drm_deinit(&drm_ctx, drm_fd);
}

int main()
{

    DetectorInit(&m_info);
    cv::VideoCapture cap("/app/4.mp4");
    cv::Mat frame;
    std::vector<StObject> st_objs;
    CTracker *tracker = new CTracker();
    std::vector<StObject> tracks;
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("./output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 15, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    cap >> frame;
    while (!frame.empty())
    {
        st_objs.clear();
        gettimeofday(&start_time, NULL);
        // 目标检测
        DetectorRun(frame, st_objs);

        for(auto& track:st_objs)
        {
        printf("tarck id:%d, clsId:%d, x:%d, age:%d, lostcnt:%d\n", track.objId, track.clsId, track.x, track.age, track.lostframe);
        }
        //--------------目标融合--------------------------------------------------------
        tracker->update(st_objs);
        tracks.clear();

        tracker->GetTracks(tracks);

        for (auto &track : tracks)
        {
            cv::Point topLeft(track.x, track.y);
            cv::Point bottomRight(track.x + track.w, track.y + track.h);
            cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, std::to_string(track.objId), cv::Point(track.x, track.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);
        }
        //----------------------------------------------------------------
        gettimeofday(&stop_time, NULL);
        printf("single frame run use %f ms\n",
               (__get_us(stop_time) - __get_us(start_time)) / 1000);
        writer.write(frame);
        // cv::imwrite("./1.jpg", frame);
        cap >> frame;
    }
    cap.release();
    writer.release();
    DetectorRelease();
}
