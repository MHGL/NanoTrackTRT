#include "track.h"
#include <chrono>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <ostream>
#include <string>

auto main(int arc, char* argv[]) -> int
{
    // 跟踪器初始化
    auto* tracker = new Track();

    // 视频加载
    std::string video_file = "../samples/test.mp4";
    cv::VideoCapture cap;
    cap.open(video_file);
    if (!cap.isOpened())
    {
        std::cerr << "Open video file failed!" << std::endl;
        return -1;
    }

    TrackResult result;

    cv::Rect roi;
    cv::Mat frame;
    bool is_tracking = false;
    cv::namedWindow("display");
    while (true)
    {
        cap.read(frame);
        if (frame.empty())
        {
            std::cerr << "Frame is empty!" << std::endl;
            break;
        }

        if (!is_tracking)
        {
            // roi = cv::selectROI("display", frame, true);
            // std::cout << "roi = " << roi << std::endl;
            roi = cv::Rect(301, 168, 32, 30);
            is_tracking = true;

            auto s_t = std::chrono::high_resolution_clock::now();
            tracker->init(frame, roi);
            auto e_t = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> d_t = e_t - s_t;
            std::cout << "tracker init time: " << d_t.count() << "sec" << std::endl;
            continue;
        }

        auto s_t = std::chrono::high_resolution_clock::now();
        tracker->update(frame);
        auto e_t = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d_t = e_t - s_t;
        std::cout << "tracker update time: " << d_t.count() << "sec" << std::endl;

        // break;
        result = tracker->getTrackResult();
        roi.x = (int)(result.center_x - result.width / 2);
        roi.y = (int)(result.center_y - result.height / 2);
        roi.width = (int)result.width;
        roi.height = (int)result.height;

        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 1);
        cv::imshow("display", frame);
        // break;

        int k = cv::waitKey(1);
        if (k == 27)
        {
            break;
            is_tracking = false;
        }
    }

    cap.release();
    delete tracker;

    return 0;
}