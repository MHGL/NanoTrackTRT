#include <cmath>

#include "utils.h"

void softmax(const cv::Mat &src, cv::Mat &dst)
{
    cv::Mat maxVal;
    cv::max(src.row(1), src.row(0), maxVal);

    src.row(1) -= maxVal;
    src.row(0) -= maxVal;

    exp(src, dst);

    cv::Mat sumVal = dst.row(0) + dst.row(1);
    dst.row(0) = dst.row(0) / sumVal;
    dst.row(1) = dst.row(1) / sumVal;
}

void elementReciprocalMax(cv::Mat &srcDst)
{
    size_t totalV = srcDst.total();
    float *ptr = srcDst.ptr<float>(0);
    for (size_t i = 0; i < totalV; i++)
    {
        float val = *(ptr + i);
        *(ptr + i) = std::max(val, 1.0f / val);
    }
}

float sizeCal(float w, float h)
{
    float pad = (w + h) * 0.5;
    float sz2 = (w + pad) * (h + pad);
    return sqrt(sz2);
}

cv::Mat sizeCal(const cv::Mat &w, const cv::Mat &h)
{
    cv::Mat pad = (w + h) * 0.5;
    cv::Mat sz2 = (w + pad).mul((h + pad));

    cv::sqrt(sz2, sz2);
    return sz2;
}

void CreatHannWindow(cv::Mat &out, int width, int height)
{
    cv::Mat vertical(height, 1, CV_32FC1);
    cv::Mat horizontal(1, width, CV_32FC1);
    for (int r = 0; r < height; r++)
    {
        vertical.at<float>(r, 0) = 0.5 - 0.5 * cos(2 * PI * r / (height - 1));
    }
    for (int c = 0; c < width; c++)
    {
        horizontal.at<float>(0, c) = 0.5 - 0.5 * cos(2 * PI * c / (width - 1));
    }

    out = vertical * horizontal;
    // return out;
}