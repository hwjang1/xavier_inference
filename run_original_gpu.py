from trt_engine import create_model_wrapper, InferenceRunner
import torch2trt
import torch
import argparse
from torch2trt import TRTModule
import cv2
from poseAction.config import cfg
from poseAction.config import update_config
from poseAction.utils.transforms import *
from poseAction import models
from tqdm import tqdm
import time
import pycuda.driver as cuda
from jtop import jtop, JtopException
import csv
import os
import subprocess

KEYPOINT_WIDTH = 224
KEYPOINT_HEIGHT = 224
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.5
DUMPING_TRT_MODEL = 'model_final_trt.pth'
DUMPING_PTH_MODEL = 'model_final.pth'
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
GWANAK_CLASS = ['Carry', 'Jest Before', 'Dumping', 'Normal']
RTSP_URLS = ['rtsp://192.168.1.180:554/stream1']
RTSP_WINDOWS_NAME = ['camera1']
RTSP_VIEWER_SOCKET_PORT = [5001]

parser = argparse.ArgumentParser()
parser.add_argument('--no_execution', help='prev Model directory', action='store_true')
parser.add_argument('--cfg', type=str, default='w32_256x256_adam_lr1e-3_multitask_coco.yaml')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
parser.add_argument('--modelDir', help='model directory', type=str, default='model_final.pth')
parser.add_argument('--logDir', help='log directory', type=str, default='')
parser.add_argument('--dataDir', help='data directory', type=str, default='')
parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

def jtop_log_write(jetson, cpu_writer, gpu_writer, power_writer, engine_writer):

    while jetson.ok():
        stats = jetson.stats
        cpu_writer.writerow(jetson.cpu)
        gpu_writer.writerow(jetson.gpu)
        power_writer.writerow(jetson.power)
        engine_writer.writerow(jetson.engine)

if __name__ == "__main__":
    args = parser.parse_args()

    names = ['dumping', 'gabbage', 'g2', 'g3']

    model_wrapper_0 = create_model_wrapper(
        model_path='/data/cctv_nas/original_model_gpu.engine',
        batch_size=1,
        device='cuda:0',
    )

    model_wrapper_1 = create_model_wrapper(
        model_path='/data/cctv_nas/original_model_gpu.engine',
        batch_size=1,
        device='cuda:0',
    )


    model_wrapper_0.load_model()
    model_wrapper_1.load_model()

    model_wrapper_0 = InferenceRunner(model_wrapper_0, '/data/cctv_nas/validation_images', './results')
    model_wrapper_1 = InferenceRunner(model_wrapper_1, '/data/cctv_nas/validation_images', './results')
    img = cv2.imread('/data/cctv_nas/validation_images/20230411122711.jpg')
    preproc_image = model_wrapper_0.preprocess_image(img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_data = torch.zeros(1, 3, 256, 256).cuda()

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    try:
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(DUMPING_TRT_MODEL))
        dumping_trt_model = model_trt.eval()
    except:
        update_config(cfg, args)
        model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False).eval().cuda()
        model.load_state_dict(torch.load(DUMPING_PTH_MODEL), strict=True)
        
        model_trt = torch2trt.torch2trt(model, [dummy_data], fp16_mode=True, max_workspace_size=1 << 25)
        torch.save(model_trt.state_dict(), DUMPING_TRT_MODEL)
        dumping_trt_model = model_trt.eval()


    cmd = ['python3', 'jtop_check.py', '--mode', 'original_gpu']
    sp = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    for i in tqdm(range(1000)):
        model_wrapper_0.model.inference_best(preproc_image)
        cuda.memcpy_dtoh(model_wrapper_0.model.outputs[0]['host_allocation'], model_wrapper_0.model.outputs[0]['allocation'])
        # dumping_trt_model(dummy_data)

        model_wrapper_1.model.inference_best(preproc_image)
        cuda.memcpy_dtoh(model_wrapper_1.model.outputs[0]['host_allocation'], model_wrapper_1.model.outputs[0]['allocation'])
        # dumping_trt_model(dummy_data)
        
        

    sp.terminate()