# NanoTrackTrt
[项目参考/模型来源](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack)    
[跟踪代码参考](https://github.com/opencv/opencv/blob/4.x/modules/video/src/tracking/tracker_nano.cpp)   

## 依赖
- opencv:4.7
- cuda:11.7
- tensorrt:8.4.1
- pybind11

## 模型转换
```bash
# 非量化模型  
cd weights
# trtexec命令在tensorrt安装包中
trtexec --onnx=nanotrack_backbone_dy_sim.onnx --saveEngine=nanotrack_backbone.engine --minShapes=input:1x3x127x127 --optShapes=input:1x3x127x127 --maxShapes=input:1x3x255x255  
trtexec --onnx=nanotrack_head_sim.onnx --saveEngine=nanotrack_head.engine
# 会在weights目录下生成nanotrack_backbone.trt模型和nanotrack_head.trt模型   

# 量化模型
cd convert
python convert.py
# 会在weights目录下生成nanotrack_backbone_int8.trt模型和nanotrack_head_int8.trt模型   

```

## 修改配置文件
```bash
# config.h
```

## 编译工程
```bash
# 修改CMakeLists的头文件设置

mkdir build 
cd build
# 编译可执行文件 会生成tracker可执行文件   
cmake ..
# 或者编译python接口 会生成tracker.cpython-310-x86_64-linux-gnu.so, 文件名字会根据当前python版本和系统架构有所变动   
cmake -DBUILD_PYTHON_LIB=ON ..

make -j8
```

## 测试
```bash
# cpp测试  
./tracker

# python测试
# 需要将test.py放到build目录下，这是因为config.h中相对目录的配置限制，如有需要，可自行修改
cp ../test.py .
python3 test.py
```

## 测试性能
- PC端 显卡NVIDIA GeForce GTX 1650  单帧延时2ms左右
- AGX Xavier 单帧延时10ms

## 下一步计划
- 经过实际测试，该算法在RK3588上运行速度为4-6ms/帧，而Xavier相差比较大    
- AGX Xavier经过INT8量化没有明显性能提升    
