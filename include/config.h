#ifndef __CONFIG_H__
#define __CONFIG_H__

namespace Config
{
namespace Model
{
// 特征提取模型路径
static const char* BACKBONE_FILE = "../weights/nanotrack_backbone_int8.trt";

// 特征融合模型路径
static const char* HEAD_FILE = "../weights/nanotrack_head_int8.trt";

// 网络步长
static const int STRIDE = 16;

// 预热数量
static const int WARMUP_NUM = 10;
}   // namespace Model

namespace Track
{
// 模板区域输入大小
static const int TEMPLATE_SIZE = 127;

// 搜索区域输入大小
static const int SEARCH_SIZE = 255;

// 上下文添加系数
static const float CONTEXT_AMOUNT = 0.5;

// hanning窗影响系数
static const float WINDOW_INFLUENCE = 0.455;

// 宽高平滑系数
static const float SCALE_LR = 0.37;

// 宽高变化惩罚系数
static const float PENALTY_K = 0.15;
}   // namespace Track
}   // namespace Config

#endif