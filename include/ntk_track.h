#ifndef __NTK_TRACK_H__
#define __NTK_TRACK_H__

#include "ntk_model.h"
#include "utils.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

using TrackResult = struct TrackResult
{
    float center_x;
    float center_y;
    float width;
    float height;
    float score;
};

class NTK_Track
{
  public:
    NTK_Track();
    ~NTK_Track();

    // track
    void init(const cv::Mat &, const cv::Rect2f &);
    void update(const cv::Mat &);

    // get result
    auto getTrackResult() -> TrackResult;

  private:
    void getSubWindow(const cv::Mat &, cv::Mat &, const int);
    void generateGrid(const int);

  private:
    // model
    NTK_Model *m_model;

    // resize
    cv::Size m_origin_size;
    float m_resize_scale;
    int m_grid_size;

    // result
    TrackResult m_track_result;

    // hanning window
    cv::Mat m_hanning_window;

    // grid
    cv::Mat m_x_grid, m_y_grid;
};

#endif