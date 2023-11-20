from trt_engine import create_model_wrapper, InferenceRunner
from yolo_nms import non_max_suppression, scale_boxes, xyxy2xywh
import torch
import cv2
import itertools
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import torchvision

names = ['dumping', 'gabbage', 'g2', 'g3']

model_wrapper_quant = create_model_wrapper(
    model_path='/data/cctv_nas/testbed_qat_noqdq.engine',
    batch_size=1,
    device='cuda:0',
)


model_wrapper_quant.load_model()


model_wrapper_quant = InferenceRunner(model_wrapper_quant, '/data/cctv_nas/validation_images', './results')

img = cv2.imread('/data/cctv_nas/validation_images/20230411122711.jpg')
preproc_image = model_wrapper_quant.preprocess_image(img)

model_wrapper_quant.model.inference_int(preproc_image)


cuda.memcpy_dtoh(model_wrapper_quant.model.outputs[3]['host_allocation'], model_wrapper_quant.model.outputs[3]['allocation'])