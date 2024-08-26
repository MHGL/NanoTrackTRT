#include "ntk_model.h"
#include "config.h"
#include <fstream>

void NTK_Logger::log(Severity severity, const char *msg) noexcept
{
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL ERROR: " << msg << std::endl;
        break;

    case Severity::kERROR:
        std::cerr << "ERROR: " << msg << std::endl;
        break;

    case Severity::kWARNING:
        std::cerr << "WARNING: " << msg << std::endl;
        break;

    case Severity::kINFO:
        std::cerr << "INFO: " << msg << std::endl;
        break;

    default:
        break;
    }
}

NTK_Model::NTK_Model()
{
    m_backbone = nullptr;
    m_head = nullptr;
}

NTK_Model::~NTK_Model()
{
    release(m_backbone);
    release(m_head);
}

void printTensorAttr(const char *name, size_t idx, size_t size,
                     nvinfer1::Dims dims)
{
    std::cout << "---> Tensor: " << name << " idx = " << idx
              << "  size = " << size << "  dims = {";
    for (int32_t i = 0; i < dims.nbDims; i++)
    {
        std::cout << dims.d[i];
        if (i == dims.nbDims - 1)
        {
            std::cout << "}" << std::endl;
        }
        else
        {
            std::cout << ", ";
        }
    }
}

void NTK_Model::load(const char *trt_file, NTK_Engine *ntk_engine)
{
    /* create cuda stream
     * *************************************************************************/
    cudaStreamCreate(&ntk_engine->m_stream);

    /* load engine
     * *******************************************************************************/
    std::cout << "---> load engine from file: " << trt_file << std::endl;

    std::fstream file(trt_file, std::ios::binary | std::ios::in);

    if (!file.good())
    {
        std::cerr << "read file error: " << trt_file << std::endl;
        return;
    }

    long size = 0;
    char *trt_stream = nullptr;

    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);

    trt_stream = new char[size];
    file.read(trt_stream, size);
    file.close();

    ntk_engine->m_engine = m_runtime->deserializeCudaEngine(trt_stream, size);
    ntk_engine->m_ctx = ntk_engine->m_engine->createExecutionContext();

    delete[] trt_stream;

    if (ntk_engine->m_engine == nullptr || ntk_engine->m_ctx == nullptr)
        return;

    /* Tensor Info
     * *****************************************************************************/
    ntk_engine->m_in_num = 0;
    ntk_engine->m_out_num = 0;
    for (int32_t i = 0; i < ntk_engine->m_engine->getNbBindings(); i++)
    {
        const char *name = ntk_engine->m_engine->getBindingName(i);

        bool is_input = ntk_engine->m_engine->bindingIsInput(i);
        if (is_input)
        {
            ntk_engine->m_in_name.emplace_back(name);
            ntk_engine->m_in_num += 1;
        }
        else
        {
            ntk_engine->m_out_name.emplace_back(name);
            ntk_engine->m_out_num += 1;
        }
    }

    std::cout << "---> model input dynamic: " << ntk_engine->m_dynamic
              << std::endl;
    std::cout << "---> model input num: " << ntk_engine->m_in_num << std::endl;
    std::cout << "---> model output num: " << ntk_engine->m_out_num
              << std::endl;

    /* Input
     * *****************************************************************************/
    ntk_engine->m_in_size.resize(ntk_engine->m_in_num);
    if (ntk_engine->m_want_host_in)
    {
        ntk_engine->m_host_in.resize(ntk_engine->m_in_num);
    }
    ntk_engine->m_device_in.resize(ntk_engine->m_in_num);

    if (ntk_engine->m_dynamic)
    {
        assert(ntk_engine->m_in_num == ntk_engine->m_in_dims.size());
    }

    for (int i = 0; i < ntk_engine->m_in_num; i++)
    {
        const char *cur_tensor = ntk_engine->m_in_name[i];

        nvinfer1::Dims cur_dims;
        if (ntk_engine->m_dynamic)
        {
            cur_dims = ntk_engine->m_in_dims[i];
            assert(getNvDimSize(cur_dims, nvinfer1::DataType::kFLOAT));
            assert(ntk_engine->m_ctx->setBindingDimensions(i, cur_dims));
        }
        else
        {
            cur_dims = ntk_engine->m_ctx->getBindingDimensions(i);
        }

        ntk_engine->m_in_size[i] =
            getNvDimSize(cur_dims, ntk_engine->m_engine->getBindingDataType(i));

        printTensorAttr(cur_tensor, i, ntk_engine->m_in_size[i], cur_dims);

        if (ntk_engine->m_want_host_in)
        {
            assert(cudaMallocHost(&(ntk_engine->m_host_in[i]),
                                  ntk_engine->m_in_size[i]) == cudaSuccess);
        }
        assert(cudaMalloc(&(ntk_engine->m_device_in[i]),
                          ntk_engine->m_in_size[i]) == cudaSuccess);

        ntk_engine->m_bindings.emplace_back(ntk_engine->m_device_in[i]);
    }

    /* Output
     * *****************************************************************************/
    ntk_engine->m_out_size.resize(ntk_engine->m_out_num);
    if (ntk_engine->m_want_host_out)
    {
        ntk_engine->m_host_out.resize(ntk_engine->m_out_num);
    }
    ntk_engine->m_device_out.resize(ntk_engine->m_out_num);

    for (int i = 0; i < ntk_engine->m_out_num; i++)
    {
        const char *cur_tensor = ntk_engine->m_out_name[i];

        int startIdx = ntk_engine->m_in_num;
        nvinfer1::Dims cur_dims =
            ntk_engine->m_ctx->getBindingDimensions(startIdx + i);
        ntk_engine->m_out_size[i] = getNvDimSize(
            cur_dims, ntk_engine->m_engine->getBindingDataType(startIdx + i));

        printTensorAttr(cur_tensor, startIdx + i, ntk_engine->m_out_size[i],
                        cur_dims);

        if (ntk_engine->m_want_host_out)
        {
            assert(cudaMallocHost(&(ntk_engine->m_host_out[i]),
                                  ntk_engine->m_out_size[i]) == cudaSuccess);
        }

        assert(cudaMalloc(&(ntk_engine->m_device_out[i]),
                          ntk_engine->m_out_size[i]) == cudaSuccess);

        ntk_engine->m_bindings.emplace_back(ntk_engine->m_device_out[i]);
    }
}

void NTK_Model::release(NTK_Engine *ntk_engine)
{
    if (ntk_engine == nullptr)
        return;

    cudaStreamDestroy(ntk_engine->m_stream);

    for (int i = 0; i < ntk_engine->m_in_num; i++)
    {
        if (ntk_engine->m_want_host_in)
            cudaFreeHost(ntk_engine->m_host_in[i]);

        cudaFree(ntk_engine->m_device_in[i]);
    }

    for (int i = 0; i < ntk_engine->m_out_num; i++)
    {
        if (ntk_engine->m_want_host_out)
            cudaFreeHost(ntk_engine->m_host_out[i]);

        cudaFree(ntk_engine->m_device_out[i]);
    }

    delete ntk_engine->m_ctx;
}

auto NTK_Model::getNvDimSize(nvinfer1::Dims dims,
                             nvinfer1::DataType type) -> size_t
{
    size_t size = 1;
    switch (type)
    {
    case nvinfer1::DataType::kINT8:
        size = 1;
        break;
    case nvinfer1::DataType::kHALF:
        size = 2;
        break;
    case nvinfer1::DataType::kFLOAT:
        size = 4;
        break;
    default:
        break;
    }

    for (int i = 0; i < dims.nbDims; i++)
    {
        size *= dims.d[i];
    }

    // 返回 size 变量的拷贝
    return size;
}

auto NTK_Model::isReady() -> bool
{
    // tensorrt runtime
    m_runtime = nvinfer1::createInferRuntime(m_logger);

    // backbone engine
    std::cout << "alloc resource for backbone: " << std::endl;

    m_backbone = new NTK_Engine();
    m_backbone->m_dynamic = true;
    m_backbone->m_in_dims = {nvinfer1::Dims{4, 1, 3, Config::Track::SEARCH_SIZE,
                                            Config::Track::SEARCH_SIZE}};
    m_backbone->m_want_host_in = true;
    m_backbone->m_want_host_out = false;
    load(Config::Model::BACKBONE_FILE, m_backbone);

    // head engine
    std::cout << "alloc resource for head: " << std::endl;

    m_head = new NTK_Engine();
    m_head->m_want_host_in = false;
    m_head->m_want_host_out = true;
    load(Config::Model::HEAD_FILE, m_head);

    return true;
}

void NTK_Model::forward(const cv::Mat &cv_img)
{
    if ((!m_backbone->m_in_num) || (!m_head->m_out_num))
    {
        std::cerr << "call isReady first!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // 数据拷贝
    int h = cv_img.rows;
    int w = cv_img.cols;
    int c = cv_img.channels();

    int size = cv_img.cols;
    assert((w == Config::Track::TEMPLATE_SIZE) ||
           (w == Config::Track::SEARCH_SIZE));

    // HWC -> CHW
    auto *src = reinterpret_cast<float *>(cv_img.data);
    int src_stride = w * c;

    auto *dst = reinterpret_cast<float *>(m_backbone->m_host_in[0]);
    int dst_stride = h * w;

    for (int _c = 0; _c < c; _c++)
    {
        for (int _h = 0; _h < h; _h++)
        {
            for (int _w = 0; _w < w; _w++)
            {
                int src_idx = _h * src_stride + _w * c + _c;
                int dst_idx = _c * dst_stride + _h * w + _w;

                dst[dst_idx] = src[src_idx];
            }
        }
    }

    // 当前输入对应的输出和head输入
    int index = (size == Config::Track::TEMPLATE_SIZE) ? 0 : 1;

    // 设置输入shape
    nvinfer1::Dims dims = nvinfer1::Dims{4, 1, 3, size, size};
    size_t dim_size =
        getNvDimSize(dims, m_backbone->m_engine->getBindingDataType(0));

    if (dim_size != m_backbone->m_in_size[0])
    {
        m_backbone->m_ctx->setBindingDimensions(0, dims);
        m_backbone->m_in_size[0] = dim_size;
    }

    cudaMemcpyAsync(m_backbone->m_device_in[0], m_backbone->m_host_in[0],
                    m_backbone->m_in_size[0], NTK_H2D, m_backbone->m_stream);

    m_backbone->m_ctx->enqueueV2(m_backbone->m_bindings.data(),
                                 m_backbone->m_stream, nullptr);

    cudaMemcpyAsync(m_head->m_device_in[index], m_backbone->m_device_out[0],
                    m_head->m_in_size[index], NTK_D2D, m_backbone->m_stream);

    // 如果index为0，那么代表跟踪器初始化，只执行backbone
    if (index == 0)
        return;

    cudaStreamSynchronize(m_backbone->m_stream);

    // head
    m_head->m_ctx->enqueueV2(m_head->m_bindings.data(), m_head->m_stream,
                             nullptr);

    for (int i = 0; i < m_head->m_out_num; i++)
    {
        cudaMemcpyAsync(m_head->m_host_out[i], m_head->m_device_out[i],
                        m_head->m_out_size[i], NTK_D2H, m_head->m_stream);
    }
}

auto NTK_Model::get() -> std::vector<cv::Mat>
{
    cudaStreamSynchronize(m_head->m_stream);

    std::vector<cv::Mat> ret;
    ret.resize(m_head->m_out_num);

    int startIdx = m_head->m_in_num;
    for (int i = 0; i < m_head->m_out_num; i++)
    {
        nvinfer1::Dims dims = m_head->m_ctx->getBindingDimensions(startIdx + i);
        int32_t c = dims.d[1];
        int32_t h = dims.d[2];
        int32_t w = dims.d[3];

        cv::Mat mat(c * h * w, 1, CV_32F,
                    reinterpret_cast<float *>(m_head->m_host_out[i]));

        ret[i] = mat.reshape(0, {c, h, w});
    }

    return ret;
}
