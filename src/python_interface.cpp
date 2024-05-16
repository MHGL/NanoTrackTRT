#include <string>

#include "opencv2/opencv.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tracker.h"  // 你的 TrackerNanoTRT 类的头文件

namespace py = pybind11;

class PyTracker {
 public:
  PyTracker(const std::string &config) {
    const YAML::Node node = YAML::Load(config);
    tracker = new TrackerNanoTRT(node);
  }

  ~PyTracker() { delete tracker; }

  void init(py::array_t<unsigned char> image,
            std::tuple<int, int, int, int> roi) {
    cv::Mat frame(image.shape(0), image.shape(1), CV_8UC3,
                  image.mutable_data());
    cv::Rect rect(std::get<0>(roi), std::get<1>(roi), std::get<2>(roi),
                  std::get<3>(roi));
    tracker->init(frame, rect);
  }

  std::tuple<int, int, int, int> update(py::array_t<unsigned char> image) {
    cv::Mat frame(image.shape(0), image.shape(1), CV_8UC3,
                  image.mutable_data());

    cv::Rect rect = tracker->update(frame);

    return std::make_tuple((int)rect.x, (int)rect.y, (int)rect.width,
                           (int)rect.height);
  }

  void setRect(float cx, float cy, float w, float h) {
    tracker->targetPos = {cx, cy};
    tracker->targetSz = {w, h};
  }

  float getScore() { return tracker->tracking_score; }

 private:
  TrackerNanoTRT* tracker;
};

PYBIND11_MODULE(tracker, m) {
  m.doc() = "pybind11 TrackerNanoTRT plugin";

  py::class_<PyTracker>(m, "Tracker")
      .def(py::init<const std::string &>())
      .def("init", &PyTracker::init, py::arg("image"), py::arg("roi"),
           "Declaration: void init(numpy.array, tuple)")
      .def("update", &PyTracker::update, py::arg("image"),
           "Declaration: tuple update(numpy.array)")
      .def("setRect", &PyTracker::setRect, py::arg("cx"), py::arg("cy"),
           py::arg("w"), py::arg("h"), "Declaration: reset tracking rect")
      .def("getScore", &PyTracker::getScore, "Declaration: get tracking score");
}