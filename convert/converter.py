# -*- coding:utf-8 -*-
import tensorrt as trt

from pathlib import Path


def build_int8_engine(onnx_file: str = None, 
                      engine_file: str = None, 
                      batch_size: int = 1,
                      dynamic_mode: bool = False,
                      dynamic_range: list = [], 
                      calibrator: object = None) -> None:
    print(">>> start convert {} to {} ... ... ".format(onnx_file, engine_file))

    onnx_file = Path(onnx_file)
    if not onnx_file.exists():
        raise FileNotFoundError("can not found onnx file: {}".format(onnx_file))
    
    engine_file = Path(engine_file)

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(G_LOGGER) as builder, \
            builder.create_network(batch_size << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, G_LOGGER) as parser:
        
        if not builder.platform_has_fast_int8:
            raise RuntimeError("not support int8")
        
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * (1 << 16))
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

        if not parser.parse(onnx_file.read_bytes()):
            for i in range(parser.num_errors):
                print(">>> onnx parser error: ", parser.get_error(i))
            raise RuntimeError("parse onnx file: {} error".format(onnx_file))
        
        input_name = network.get_input(0).name
        if dynamic_mode:
            profile = builder.create_optimization_profile()
            profile.set_shape(input_name, *dynamic_range)
            config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("build engine error")
        
        engine_file.write_bytes(engine)

        print(">>> build engine done!")

        
if __name__ == "__main__":
    from calibrator import Calibrator

    onnx_file = "../weights/nanotrack_backbone_dy_sim.onnx"
    engine_file = "../weights/nanotrack_backbone_int8.trt"
    calib = Calibrator(dataset_file="../weights/backbone_quant_inputs.txt",
                       cache_file="./backbone.cache",
                       batch_size=1) 

    build_int8_engine(onnx_file=onnx_file,
                      engine_file=engine_file,
                      batch_size=1,
                      dynamic_mode=True,
                      dynamic_range=[[1, 3, 127, 127], [1, 3, 255, 255], [1, 3, 255, 255]],
                      calibrator=calib)
    
    onnx_file = "../weights/nanotrack_head_sim.onnx"
    engine_file = "../weights/nanotrack_head_int8.trt"
    calib = Calibrator(dataset_file="../weights/head_quant_inputs.txt",
                       cache_file="./head.cache",
                       batch_size=1)
    
    build_int8_engine(onnx_file=onnx_file,
                      engine_file=engine_file,
                      batch_size=1,
                      dynamic_mode=False,
                      dynamic_range=[],
                      calibrator=calib)