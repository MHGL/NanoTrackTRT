# -*- coding:utf-8 -*-
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from typing import Union
from pathlib import Path


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataset_file: str = None, cache_file: str = None, batch_size: int = 1) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)

        # dataset
        dataset = Path(dataset_file)
        if not dataset.exists():
            raise FileNotFoundError("dataset file not found!")
        
        lines = dataset.read_text()
        self.dataset = [list(map(lambda x: dataset.parent.joinpath(x.strip()), i.split(','))) 
                            for i in lines.split('\n') if i]

        # cache
        self.cache = Path(cache_file)

        # batch index
        self.batch_index = 0

        # batch_size
        self.batch_size = batch_size

        # max batch num
        self.max_batch_num = len(self.dataset) // self.batch_size

        # cuda memory
        self.valid_suffix = [".jpg", ".png", ".jpeg", ".npy"]

        self.device_inputs = list()
        for data in self.dataset[0]:
            data = Path(data)
            
            if data.suffix not in self.valid_suffix:
                raise TypeError("expected suffix in {}".format(self.valid_suffix))
            
            if data.suffix == self.valid_suffix[-1]:
                arr = np.load(str(data))
                size = arr.size * self.batch_size * trt.float32.itemsize
            else:
                cv_img = cv2.imread(str(data))
                size = cv_img.size * self.batch_size * trt.float32.itemsize

            self.device_inputs.append(cuda.mem_alloc(size))

    def next_batch(self) -> list:
        batch = list()
        
        if self.batch_index >= self.max_batch_num:
            return batch
        
        for data_files in self.dataset[self.batch_index: self.batch_index + self.batch_size]:
            for i, data_file in enumerate(data_files):
                if data_file.suffix == self.valid_suffix[-1]:
                    img = np.load(str(data_file))
                    
                else:
                    cv_img = cv2.imread(str(data_file))
                    img = cv_img.transpose(2, 0, 1)
                    img = img[None, ...]

                img = img.astype(np.float32)

                if len(batch) > i:
                    batch[i] = np.concatenate([batch[i], img], axis=0)
                else:
                    batch.append(img)

        self.batch_index += 1

        return batch

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: str, p_str: str = None) -> Union[None, int]:
        batch = self.next_batch()
        if not batch:
            return None
        
        for d, h in zip(self.device_inputs, batch):
            cuda.memcpy_htod(d, np.ascontiguousarray(h))

        return [int(d) for d in self.device_inputs]

    def read_calibration_cache(self) -> Union[None, bytes]:
        if self.cache.exists():
            return self.cache.read_bytes()
        
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache.write_bytes(cache)


if __name__ == "__main__":
    c = Calibrator(dataset_file="../weights/backbone_quant_inputs.txt", cache_file="./.cache", batch_size=1)
    d = c.next_batch()
    print(">>> d: ", len(d))
    print(">>> d: ", [i.shape for i in d])