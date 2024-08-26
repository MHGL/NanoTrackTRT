cmake_minimum_required(VERSION 3.10)
project(tracker)

set(CMAKE_CXX_STANDARD 11)

# cmake params
option(BUILD_PYTHON_LIB "build for python interface" OFF)

# opencv deps
set(OpenCV_DIR /opt/opencv_4.7.0_x86_64/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# project headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda deps
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt deps
include_directories(/opt/tensorrt_8.4.1.5_x86_64/include)
link_directories(/opt/tensorrt_8.4.1.5_x86_64/lib)

if (BUILD_PYTHON_LIB)
    message("BUILD_PYTHON_LIB is enabled")

    # pybind11 
    set(PYBIND11_DIR /opt/pybind11_2.13.1_x86_64)
    find_package(pybind11 REQUIRED PATHS ${PYBIND11_DIR})

    # source
    set(SOURCE_FILE ${PROJECT_SOURCE_DIR}/src/ntk_model.cpp
                    ${PROJECT_SOURCE_DIR}/src/ntk_track.cpp  
                    ${PROJECT_SOURCE_DIR}/src/utils.cpp 
                    ${PROJECT_SOURCE_DIR}/src/python_interface.cpp)    
                    
    pybind11_add_module(tracker ${SOURCE_FILE})

    set_target_properties(tracker PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

else()
    # source
    set(SOURCE_FILE ${PROJECT_SOURCE_DIR}/test.cpp
                    ${PROJECT_SOURCE_DIR}/src/utils.cpp
                    ${PROJECT_SOURCE_DIR}/src/ntk_model.cpp   
                    ${PROJECT_SOURCE_DIR}/src/ntk_track.cpp)    
                    
    add_executable(tracker ${SOURCE_FILE})

endif()

target_link_libraries(tracker PUBLIC ${CUDA_LIBS}
                                ${OpenCV_LIBS}
                                cudart
                                nvinfer)    
                                