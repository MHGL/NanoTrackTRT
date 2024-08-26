#include "ntk_track.h"
#include "config.h"
#include "ntk_model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <string>
#include <vector>

NTK_Track::NTK_Track()
{
    // 模型初始化
    m_model = new NTK_Model();
    if (!m_model->isReady())
    {
        std::cerr << "init model error" << std::endl;
        return;
    }

    // 结果初始化
    memset(&m_track_result, 0.0, sizeof(m_track_result));
}

NTK_Track::~NTK_Track()
{
    // 释放资源
    delete m_model;
}

void NTK_Track::generateGrid(const int size)
{
    int step = size / 2;

    std::vector<float> grid_1d(size, 0);
    for (int i = 0; i < size; i++)
    {
        grid_1d[i] = (float)(i - step);
    }

    cv::Mat mat_1d(1, size, CV_32FC1, grid_1d.data());
    cv::repeat(mat_1d, size, 1, m_x_grid);
    cv::repeat(mat_1d.t(), 1, size, m_y_grid);

    m_x_grid *= Config::Model::STRIDE;
    m_y_grid *= Config::Model::STRIDE;
}

void NTK_Track::getSubWindow(const cv::Mat &src, cv::Mat &dst,
                             const int window_size)
{
    const cv::Scalar avg_channels = cv::mean(src);
    const cv::Size src_size = src.size();

    float center_x = m_track_result.center_x;
    float center_y = m_track_result.center_y;
    float width = m_track_result.width;
    float height = m_track_result.height;

    float context_sz = (width + height) * Config::Track::CONTEXT_AMOUNT;
    float sz = sqrt((width + context_sz) * (height + context_sz));

    // 特征图尺度和127/255相对应，需要scale参数返回原图尺度
    m_resize_scale = Config::Track::TEMPLATE_SIZE / sz;

    // 基于搜索区域和目标区域比例来确定crop size
    float crop_sz =
        sz * static_cast<float>(window_size) / Config::Track::TEMPLATE_SIZE;
    float c = (crop_sz + 1) / 2;
    int x1 = floor(center_x - c + 0.5);
    int x2 = x1 + crop_sz - 1;
    int y1 = floor(center_y - c + 0.5);
    int y2 = y1 + crop_sz - 1;

    const int left_pad = std::max(0, -x1);
    const int top_pad = std::max(0, -y1);
    const int right_pad = std::max(0, x2 - src_size.width + 1);
    const int bottom_pad = std::max(0, y2 - src_size.height + 1);

    x1 += left_pad;
    x2 += left_pad;
    y1 += top_pad;
    y2 += top_pad;

    dst = src;
    if (left_pad || right_pad || top_pad || bottom_pad)
    {
        cv::copyMakeBorder(src, dst, top_pad, bottom_pad, left_pad, right_pad,
                           cv::BORDER_CONSTANT, avg_channels);
    }

    dst = dst(cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
    cv::resize(dst, dst, cv::Size(window_size, window_size), 0, 0,
               cv::INTER_LINEAR);
    dst.convertTo(dst, CV_32F, 1.0, 0.0);
}

void NTK_Track::init(const cv::Mat &cv_img_, const cv::Rect2f &roi)
{
    m_grid_size = (Config::Track::SEARCH_SIZE - Config::Track::TEMPLATE_SIZE) /
                      Config::Model::STRIDE +
                  8;

    cv::Mat cv_img = cv_img_.clone();
    m_origin_size = cv_img.size();

    // track result 赋值
    m_track_result.center_x = roi.x + (roi.width - 1) / 2;
    m_track_result.center_y = roi.y + (roi.height - 1) / 2;
    m_track_result.width = roi.width;
    m_track_result.height = roi.height;

    cv::Mat input;
    getSubWindow(cv_img, input, Config::Track::TEMPLATE_SIZE);

    m_model->forward(input);

    CreatHannWindow(m_hanning_window, m_grid_size, m_grid_size);
    generateGrid(m_grid_size);
}

void NTK_Track::update(const cv::Mat &cv_img_)
{
    cv::Mat cv_img = cv_img_.clone();

    cv::Mat input;
    getSubWindow(cv_img, input, Config::Track::SEARCH_SIZE);

    m_model->forward(input);

    std::vector<cv::Mat> maps = m_model->get();
    cv::Mat score_map = maps[0];
    cv::Mat bbox_map = maps[1];

    cv::Mat score_softmax;
    softmax(score_map, score_softmax);
    cv::Mat score = score_softmax.row(1);
    score = score.reshape(0, {m_grid_size, m_grid_size});

    cv::Mat pred_x1 =
        m_x_grid - bbox_map.row(0).reshape(0, {m_grid_size, m_grid_size});
    cv::Mat pred_y1 =
        m_y_grid - bbox_map.row(1).reshape(0, {m_grid_size, m_grid_size});
    cv::Mat pred_x2 =
        m_x_grid + bbox_map.row(2).reshape(0, {m_grid_size, m_grid_size});
    cv::Mat pred_y2 =
        m_y_grid + bbox_map.row(3).reshape(0, {m_grid_size, m_grid_size});

    cv::Mat pred_cx = (pred_x1 + pred_x2) * 0.5;
    cv::Mat pred_cy = (pred_y1 + pred_y2) * 0.5;
    cv::Mat pred_w = pred_x2 - pred_x1;
    cv::Mat pred_h = pred_y2 - pred_y1;

    // scale penalty
    cv::Mat sc = sizeCal(pred_w, pred_h) /
                 sizeCal(m_track_result.width * m_resize_scale,
                         m_track_result.height * m_resize_scale);
    elementReciprocalMax(sc);

    // aspect ratio penalty
    float ratio = m_track_result.width / m_track_result.height;
    cv::Mat rc(m_grid_size, m_grid_size, CV_32FC1, cv::Scalar::all(ratio));
    rc /= (pred_w / pred_h);
    elementReciprocalMax(rc);

    cv::Mat penalty;
    cv::exp(((rc.mul(sc) - 1) * Config::Track::PENALTY_K * (-1)), penalty);

    cv::Mat pscore = penalty.mul(score);
    float window_influence = Config::Track::WINDOW_INFLUENCE;
    pscore =
        pscore * (1 - window_influence) + m_hanning_window * window_influence;

    int best_idx[2] = {0, 0};
    cv::minMaxIdx(pscore, 0, 0, 0, best_idx);

    float cx = pred_cx.at<float>(best_idx) / m_resize_scale;
    float cy = pred_cy.at<float>(best_idx) / m_resize_scale;
    float w = pred_w.at<float>(best_idx) / m_resize_scale;
    float h = pred_h.at<float>(best_idx) / m_resize_scale;

    float lr = penalty.at<float>(best_idx) * score.at<float>(best_idx) *
               Config::Track::SCALE_LR;

    cx += m_track_result.center_x;
    cy += m_track_result.center_y;
    w = w * lr + (1 - lr) * m_track_result.width;
    h = h * lr + (1 - lr) * m_track_result.height;

    m_track_result.score = score.at<float>(best_idx);
    m_track_result.center_x =
        std::max(0.f, std::min(cx, (float)m_origin_size.width));
    m_track_result.center_y =
        std::max(0.f, std::min(cy, (float)m_origin_size.height));
    m_track_result.width =
        std::max(10.f, std::min(w, (float)m_origin_size.width));
    m_track_result.height =
        std::max(10.f, std::min(h, (float)m_origin_size.height));
}

auto NTK_Track::getTrackResult() -> TrackResult
{
    return m_track_result;
}
