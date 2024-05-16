#ifndef __UTILS_H__
#define __UTILS_H__

#include "opencv2/opencv.hpp"

void softmax(const cv::Mat &src, cv::Mat &dst);

void elementReciprocalMax(cv::Mat &srcDst);

float sizeCal(float w, float h);

cv::Mat sizeCal(const cv::Mat &w, const cv::Mat &h);

#define PI 3.1415926
void CreatHannWindow(cv::Mat &out, int width, int height);

#endif