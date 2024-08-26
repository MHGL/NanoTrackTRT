#include "ntk_track.h"
#include "opencv2/opencv.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

class PyTracker
{
  public:
    PyTracker() { tracker = new NTK_Track(); }

    ~PyTracker() { delete tracker; }

    void init(py::array_t<unsigned char> image,
              std::tuple<int, int, int, int> roi)
    {
        cv::Mat frame(image.shape(0), image.shape(1), CV_8UC3,
                      image.mutable_data());
        cv::Rect rect(std::get<0>(roi), std::get<1>(roi), std::get<2>(roi),
                      std::get<3>(roi));
        tracker->init(frame, rect);
    }

    std::tuple<int, int, int, int, float>
    update(py::array_t<unsigned char> image)
    {
        cv::Mat frame(image.shape(0), image.shape(1), CV_8UC3,
                      image.mutable_data());

        tracker->update(frame);
        TrackResult ret = tracker->getTrackResult();

        return std::make_tuple((int)ret.center_x, (int)ret.center_y,
                               (int)ret.width, (int)ret.height,
                               (float)ret.score);
    }

  private:
    NTK_Track *tracker;
};

PYBIND11_MODULE(tracker, m)
{
    m.doc() = "pybind11 TrackerNanoTRT plugin";

    py::class_<PyTracker>(m, "Tracker")
        .def(py::init<>())
        .def("init", &PyTracker::init, py::arg("image"), py::arg("roi"),
             "Declaration: void init(numpy.array, tuple)")
        .def("update", &PyTracker::update, py::arg("image"),
             "Declaration: tuple update(numpy.array)");
}