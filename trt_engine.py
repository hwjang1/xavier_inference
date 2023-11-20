import cv2
import numpy as np
import time
import argparse
import yaml
import os
import sys
from pathlib import Path
import itertools

# import lib for tensorrt
try:
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
    import torch
    import torchvision
except ImportError:
    print("Failed to load tensorrt, pycuda")
    trt = None
    cuda = None



class TRTWrapper():
    """TensorRT model wrapper."""
    def __init__(self, model_path, batch):
        self.model_path = model_path
        self._batch = batch
        self._bindings = None

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value

    @property
    def bindings(self):
        return self._bindings

    @bindings.setter
    def bindings(self, value):
        self._bindings = value

    def load_model(self, DLA_core=None):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER) # serialized ICudEngine을 deserialized하기 위한 클래스 객체
        if DLA_core:
            runtime.DLA_core = DLA_core
        trt.init_libnvinfer_plugins(None, "") # plugin 사용을 위함
        with open(self.model_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read()) # trt 모델을 읽어 serialized ICudEngine을 deserialized함
        
        self.context = self.engine.create_execution_context() # ICudEngine을 이용해 inference를 실행하기 위한 context class생        assert self.engine 
        assert self.context
        
        self.alloc_buf()

    def inference(self, input_image):
        # print(input_image.shape)
        image = input_image.transpose(0, 3, 1, 2) # NHWC to NWHC
        # print(image.shape)
        image = np.ascontiguousarray(image) 
        cuda.memcpy_htod(self.inputs[0]['allocation'], image) # input image array(host)를 GPU(device)로 보내주는 작업
        self.context.execute_v2(self.allocations) #inference 실행!
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation']) # GPU에서 작업한 값을 host로 보냄
        

        # print(self.outputs[3])
        
        # num_detections = self.outputs[0]['host_allocation'] # detection된 object개수
        # nmsed_boxes = self.outputs[1]['host_allocation'] # detection된 object coordinate
        # nmsed_scores = self.outputs[2]['host_allocation'] # detection된 object confidence
        # nmsed_classes = self.outputs[3]['host_allocation'] # detection된 object class number
        # result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        # return result
        return self.outputs[3]['host_allocation']

    def inference_best(self, input_image):
        # print(input_image.shape)
        image = input_image.transpose(0, 3, 1, 2) # NHWC to NWHC
        # print(image.shape)
        image = np.ascontiguousarray(image) 
        cuda.memcpy_htod(self.inputs[0]['allocation'], image) # input image array(host)를 GPU(device)로 보내주는 작업
        self.context.execute_v2(self.allocations) #inference 실행!
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation']) # GPU에서 작업한 값을 host로 보냄
        

        # print(self.outputs[3])
        
        # num_detections = self.outputs[0]['host_allocation'] # detection된 object개수
        # nmsed_boxes = self.outputs[1]['host_allocation'] # detection된 object coordinate
        # nmsed_scores = self.outputs[2]['host_allocation'] # detection된 object confidence
        # nmsed_classes = self.outputs[3]['host_allocation'] # detection된 object class number
        # result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        # return result
        return self.outputs[0]['host_allocation']
        
    def inference_int(self, input_image):
        # print(input_image.shape)
        image = input_image.transpose(0, 3, 1, 2) # NHWC to NWHC
        # print(image.shape)
        image = np.ascontiguousarray(image) 
        image = image.astype(np.int8)
        cuda.memcpy_htod(self.inputs[0]['allocation'], image) # input image array(host)를 GPU(device)로 보내주는 작업
        self.context.execute_v2(self.allocations) #inference 실행!
        # for o in range(len(self.outputs)):
        #     cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation']) # GPU에서 작업한 값을 host로 보냄
        

        # # print(self.outputs[3])
        
        # # num_detections = self.outputs[0]['host_allocation'] # detection된 object개수
        # # nmsed_boxes = self.outputs[1]['host_allocation'] # detection된 object coordinate
        # # nmsed_scores = self.outputs[2]['host_allocation'] # detection된 object confidence
        # # nmsed_classes = self.outputs[3]['host_allocation'] # detection된 object class number
        # # result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        # # return result
        # return self.outputs[3]['host_allocation']



    def alloc_buf(self):
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []

        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i): # i번째 binding이 input인지 확인
                is_input = True 
            name = self.engine.get_binding_name(i) # i번째 binding의 이름
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i))) # i번째 binding의 data type
            # print(name, dtype)
            shape = self.context.get_binding_shape(i) # i번째 binding의 shape

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize # data type의 bit수
            for s in shape:
                size *= s # data type * 각 shape(e.g input의 경우 [1,3,640,640]) element 을 곱하여 size에 할당

            allocation = cuda.mem_alloc(size) # 해당 size만큼의 GPU memory allocation함
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i): # binding이 input이면
                self.inputs.append(binding)
            else: # 아니면 binding은 모두 output임
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))        

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs


def create_model_wrapper(model_path: str, batch_size: int, device: str):
    """Create model wrapper class."""
    assert trt and cuda, f"Loading TensorRT, Pycuda lib failed."
    model_wrapper = TRTWrapper(model_path, batch_size)

    return model_wrapper

class Colors():
    """Color class."""
    def __init__(self):
        hex = ('B55151', 'FF3636', 'FF36A2', 'CB72A2', 'EC3AFF', '3B1CFF', '7261E1', '6991BF', '00B1BD', '00BD8B',
               '00DA33', 'BEEF4D', '8B8B8B', 'FFB300', '7F5903', '411C06', '795454', '495783', '624F70', '7A7D62')
        self.palette = [self.hextorgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, m, bgr=False):
        c = self.palette[int(m) % self.n]
        return (c[2], c[1], c[0])

    @staticmethod
    def hextorgb(hex):
        return tuple(int(hex[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class InferenceRunner():
    """Inference Runner."""
    def __init__(
        self,
        model_wrapper: TRTWrapper,
        img_folder: str,
        save_dir: str
    ):
        self.model = model_wrapper
        self.img_folder = img_folder
        self.save_dir = save_dir

    def run(self, iterations=200):
        for i, filename in enumerate(os.listdir(self.img_folder)):
            # save path
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / filename
            # load image
            img = cv2.imread(os.path.join(self.img_folder, filename))
            # if image load failed
            if img is None:
                continue
            preproc_image = self.preprocess_image(img)
            # inference
            inf_res = self.model.inference(preproc_image)
            inf_res = torch.from_numpy(inf_res)
            nms_inf_res = self.non_max_suppression(inf_res)
            nmsed_boxes = [np.array(nms_inf_res[0][0:, 0:4])]
            nmsed_scores = [np.array(list(itertools.chain(*nms_inf_res[0][0:, 4:5])))]
            nmsed_classes = [np.array(list(itertools.chain(*nms_inf_res[0][0:, 5:6]))).astype(np.int8)]
            
            self.print_result(preproc_image, (len(nms_inf_res[0]), nmsed_boxes, nmsed_scores, nmsed_classes), i, save_path)

            if i > 100:
                break

        times = []
        for i in range(20):  # GPU warmup iterations
            self.model.inference(preproc_image)
        for i in range(iterations):
            start = time.time()
            self.model.inference(preproc_image)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(
            1000 * np.average(times)))
        print("Average Throughput: {:.1f} ips".format(
            1 / np.average(times)))
            
    def run2(self, iterations=200):
        for i, filename in enumerate(os.listdir(self.img_folder)):
            # save path
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / filename
            # load image
            img = cv2.imread(os.path.join(self.img_folder, filename))
            # if image load failed
            if img is None:
                continue
            preproc_image = self.preprocess_image(img)
            # inference
            inf_res = self.model.inference_best(preproc_image)
            inf_res = torch.from_numpy(inf_res)
            nms_inf_res = self.non_max_suppression(inf_res)
            nmsed_boxes = [np.array(nms_inf_res[0][0:, 0:4])]
            nmsed_scores = [np.array(list(itertools.chain(*nms_inf_res[0][0:, 4:5])))]
            nmsed_classes = [np.array(list(itertools.chain(*nms_inf_res[0][0:, 5:6]))).astype(np.int8)]
            
            self.print_result(preproc_image, (len(nms_inf_res[0]), nmsed_boxes, nmsed_scores, nmsed_classes), i, save_path)

            if i > 100:
                break

        times = []
        for i in range(20):  # GPU warmup iterations
            self.model.inference_best(preproc_image)
        for i in range(iterations):
            start = time.time()
            self.model.inference_best(preproc_image)
            times.append(time.time() - start)
            print("Iteration {} / {}".format(i + 1, iterations), end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print("Average Latency: {:.3f} ms".format(
            1000 * np.average(times)))
        print("Average Throughput: {:.1f} ips".format(
            1 / np.average(times)))

    def preprocess_image(self, raw_bgr_image):
        """
        Description:
            Converting BGR image to RGB,
            Resizing and padding it to target size,
            Normalizing to [0, 1]
            Transforming to NCHW format
        Argument:
            raw_bgr_image: a numpy array from cv2 (BGR) (H, W, C)
        Return:
            preprocessed_image: preprocessed image (1, C, resized_H, resized_W)
            original_image: the original image (H, W, C)
            origin_h: height of the original image
            origin_w: width of the origianl image
        """

        input_size = self.model.input_spec()[0][-2:] # h,w = 640,640
        original_image = raw_bgr_image
        origin_h, origin_w, origin_c = original_image.shape
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # Calculate width and height and paddings
        r_w = input_size[1] / origin_w
        r_h = input_size[0] / origin_h
        if r_h > r_w:
            tw = input_size[1]
            th = int(r_w *  origin_h)
            tx1 = tx2 = 0
            ty1 = int((input_size[0] - th) / 2)
            ty2 = input_size[0] - th - ty1
        else:
            tw = int(r_h * origin_w)
            th = input_size[0]
            tx1 = int((input_size[1] - tw) / 2)
            tx2 = input_size[1] - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        
        # Normalize to [0,1]
        image /= 255.0
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        preprocessed_image = np.ascontiguousarray(image)
        return preprocessed_image

    def print_result(self, result_image, result_label, count, save_path):
        classes = ['person', 'gabbage_1', 'gabbage_2', 'gabbage_3']
        num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = result_label
        
        colors = Colors()
        h, w = result_image.shape[1:3]
        result_image = np.squeeze(result_image)
        result_image *= 255
        result_image = result_image.astype(np.uint8)
        # print("--------------------------------------------------------------")
        for i in range(int(num_detections)):
            detected = str(classes[int(nmsed_classes[0][i])])
            confidence_str = str(nmsed_scores[0][i])
            # unnormalize depending on the visualizing image size
            x1 = int(nmsed_boxes[0][i][0])
            y1 = int(nmsed_boxes[0][i][1])
            x2 = int(nmsed_boxes[0][i][2])
            y2 = int(nmsed_boxes[0][i][3])
            color = colors(int(nmsed_classes[0][i]), True)
            result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            text_size, _ = cv2.getTextSize(str(detected), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            result_image = cv2.rectangle(result_image, (x1, y1-5-text_h), (x1+text_w, y1), color, -1)
            result_image = cv2.putText(result_image, str(detected), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # print("Detect " + str(i+1) + "(" + str(detected) + ")")
            # print("Coordinates : [{:d}, {:d}, {:d}, {:d}]".format(x1, y1, x2, y2))
            # print("Confidence : {:.7f}".format(nmsed_scores[0][i]))
            # print("")
        # print("--------------------------------------------------------------\n\n")
        cv2.imwrite(str(save_path), cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
    def non_max_suppression(self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nm=0,  # number of masks
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output
    
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y