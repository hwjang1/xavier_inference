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
import csv
from datetime import datetime


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



if __name__ == "__main__":
    args = parser.parse_args()

    execution = not args.no_execution
    print("==============================================================")
    print(execution)
    # execution = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_data = torch.zeros(1, 3, 256, 256).cuda()

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

    time_logger = open(f'./logs/pose_time.csv', 'w', encoding='utf-8')
    time_writer = csv.DictWriter(time_logger, fieldnames=['idx', 'time'])
    time_writer.writeheader()
    time_writer.writerow({'idx': 0, 'time': datetime.now()})
    time_idx = 1
    
    for i in tqdm(range(20000)):
        dumping_trt_model(dummy_data)
        time_writer.writerow({'idx': time_idx, 'time': datetime.now()})
        time_idx += 1
        time.sleep(0.05)
        