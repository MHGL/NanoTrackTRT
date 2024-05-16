#ifndef __MODEL_H__
#define __MODEL_H__

#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <vector>

#define NTK_H2H cudaMemcpyHostToHost
#define NTK_H2D cudaMemcpyHostToDevice
#define NTK_D2H cudaMemcpyDeviceToHost
#define NTK_D2D cudaMemcpyDeviceToDevice

class NTK_Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override;
};

using NTK_Engine = struct NTK_Engine
{
    cudaStream_t m_stream;

    nvinfer1::ICudaEngine* m_engine;
    nvinfer1::IExecutionContext* m_ctx;

    bool m_dynamic;
    std::vector<nvinfer1::Dims> m_in_dims;

    int m_in_num;
    std::vector<size_t> m_in_size;
    std::vector<const char*> m_in_name;

    bool m_want_host_in;
    std::vector<void*> m_host_in;
    std::vector<void*> m_device_in;

    int m_out_num;
    std::vector<size_t> m_out_size;
    std::vector<const char*> m_out_name;

    bool m_want_host_out;
    std::vector<void*> m_host_out;
    std::vector<void*> m_device_out;

    // for tensorrt < 8.5
    std::vector<void*> m_bindings;
};

class NTK_Model
{
public:
    NTK_Model();
    ~NTK_Model();

    /*
     * initialize resource for NTK_Model
     * @return: check NTK_Model instance is ready to forward or not
     */
    auto isReady() -> bool;

    // 模型推理
    void forward(const cv::Mat& cv_img);

    // 结果获取
    auto get() -> std::vector<cv::Mat>;

private:
    /*
     * create NTK_Engine instance from trt_file
     * @param trt_file     : path of tensorrt engine file
     * @param want_float_in: the type of input is float or not
     * @param want_host_in : alloc pinned memory for input on host or not
     * @param want_host_out: alloc pinned memory for output on host or not
     * @return: the ptr of NTK_Engine instance
     */
    void load(const char* trt_file, NTK_Engine* ntk_engine);

    /*
     * release NTK_Engine resource
     * @param m_engine: ptr of NTK_Engine instance
     */
    void release(NTK_Engine* ntk_engine);

    // 辅助函数：获取dims size
    auto getNvDimSize(nvinfer1::Dims dims, nvinfer1::DataType type) -> size_t;

private:
    NTK_Logger m_logger;
    nvinfer1::IRuntime* m_runtime;

    // 模型结构体
    NTK_Engine* m_backbone;
    NTK_Engine* m_head;
};

#endif
