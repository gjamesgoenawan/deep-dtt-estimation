import tensorrt as trt
import numpy as np
import time
import cv2
import torch
import math
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
from collections import OrderedDict, namedtuple


class TensorRTEngine:
    def __init__(self, engine_path='models/object-detector/y7_b12.trt', device='cuda:0'):
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.Binding = namedtuple(
            'Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = trt.nptype(self.model.get_binding_dtype(index))
            shape = tuple(self.model.get_binding_shape(index))
            data = torch.from_numpy(
                np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.bindings[name] = self.Binding(
                name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr)
                                         for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def infer(self, img):
        self.binding_addrs['images'] = int(img.to(self.device).data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data
        return (nums, boxes, scores, classes)


class auto_segmentation():
    def __init__(self, img_size=(1920, 1280), kernel_size=640, num_camera=2, engine_path='models/object-detector/y7_b12.trt', device=torch.device('cuda:0')):
        self.img_size = img_size
        self.num_camera = num_camera
        self.kernel_size = kernel_size
        self.device = device
        self.n_tile_v = math.ceil(img_size[1]/kernel_size)
        self.n_tile_h = math.ceil(img_size[0]/kernel_size)
        self.resize_height = self.n_tile_v * self.kernel_size
        self.resize_width = self.n_tile_h * self.kernel_size

        self.origins = torch.tensor([(j*640, i*640, j*640, i*640,) for i in range(0, self.n_tile_v)
                                     for j in range(0, self.n_tile_h)] * self.num_camera, device=self.device)
        self.pred_batch = TensorRTEngine(engine_path=engine_path)

    def split_image(self, batch_img):
        """Support of cameras with different resolution is planend."""
        batch, camera_count, __, __, channel = batch_img.shape
        return torch.permute(batch_img.reshape(batch, camera_count*self.n_tile_v, self.kernel_size, self.n_tile_h, self.kernel_size, channel), (0, 1, 3, 5, 2, 4)).ravel().reshape(batch, -1)

    def get_box(self, batch_img, prev_box=None):
        if len(batch_img.shape) == 4:
            batch_img = np.expand_dims(batch_img, axis=0)
        n_batch = batch_img.shape[0]
        num_camera = batch_img.shape[1]

        if num_camera != self.num_camera:
            raise ValueError(
                f"Number of Camera is invalid. Required = {self.num_camera}, Provided = {num_camera}.\n Note that images provided must be in the shape of (batch, {self.num_camera}, {self.resize_height}, {self.resize_width}, 3)")

        if prev_box is None or (isinstance(prev_box, np.ndarray) and (prev_box == None).all()) or (isinstance(prev_box, list) and prev_box == None):
            replace_missing = False
        else:
            replace_missing = True
            if len(prev_box) != num_camera and not (isinstance(prev_box[0], list) or isinstance(prev_box[0], np.ndarray)):
                raise ValueError(
                    f"Length of Previous Box has to match the number of cameras. prev_box length = {len(prev_box)}, number of cameras = {num_camera}")

        if batch_img.shape[1:] != (self.num_camera, self.resize_height, self.resize_width, 3):
            raise ValueError(
                f"Images must be in the shape of (batch, {self.num_camera}, {self.resize_height}, {self.resize_width}, 3). Given: {batch_img.shape}")

        split_imgs = self.split_image(batch_img)

        batch_centroids = []
        for batch in range(n_batch):
            p = self.pred_batch.infer(split_imgs[batch])
            p_box = p[1].reshape(self.num_camera, -1)
            p_class = p[3].reshape(self.num_camera, -1)
            centroids = []
            for camera in range(0, num_camera):
                x1, y1, x2, y2 = 99999, 99999, -99999, -99999
                num_detections = p_class[camera][p_class[camera] == 4].shape[0]

                # Getting Centroid from all detections < 1 ms
                if not num_detections > 0:
                    if replace_missing:
                        x_center, y_center = prev_box[camera]
                    else:
                        centroids.append([np.nan, np.nan])
                        continue
                else:
                    for j in torch.where(p_class[camera] == 4)[0]:
                        origin_index = int(j/100)
                        det_box = p_box[camera][j*4:j*4+4] + \
                            self.origins[origin_index]
                        x1 = min(x1, det_box[0])
                        y1 = min(y1, det_box[1])
                        x2 = max(x2, det_box[2])
                        y2 = max(y2, det_box[3])
                    x_center, y_center = int((x1+x2)/2), int((y1+y2)/2)

                # Clipping
                if x_center > 1600:
                    x_center = 1600
                elif x_center < 320:
                    x_center = 320
                if y_center > 760:
                    y_center = 760
                elif y_center < 320:
                    y_center = 320

                if replace_missing:
                    prev_box[camera][0] = x_center
                    prev_box[camera][1] = y_center

                centroids.append([x_center, y_center])
            batch_centroids.append(centroids)
        return np.array(batch_centroids)

    def vis_current_box(self, batch_img, batch_centroids, fig=None, ax=None):
        if fig == None or ax == None:
            fig, ax = plt.subplots(batch_img.shape[0], 2, figsize=(20, 20))
        ax = ax.reshape(-1)
        for i in range(batch_img.shape[0]):
            for j in range(batch_img.shape[1]):
                ax[i*batch_img.shape[1] + j].imshow(batch_img[i][j][batch_centroids[i][j][1]-320:batch_centroids[i]
                                                    [j][1]+320, batch_centroids[i][j][0]-320:batch_centroids[i][j][0]+320])
        return fig, ax

    def prepare_imgs(self, batch_img, prev_box=None):
        if len(batch_img.shape) == 4:
            batch_img = np.expand_dims(batch_img, axis=0)

        batch_size = batch_img.shape[0]
        num_camera = batch_img.shape[1]

        if num_camera != self.num_camera:
            raise ValueError(
                f"Number of Camera is invalid. Required = {self.num_camera}, Provided = {num_camera}.\n Note that images provided must be in the shape of (batch, {self.num_camera}, {self.resize_height}, {self.resize_width}, 3)")

        if batch_img.shape[1:] != (self.num_camera, self.resize_height, self.resize_width, 3):
            raise ValueError(
                f"Images must be in the shape of (batch, {self.num_camera}, {self.resize_height}, {self.resize_width}, 3). Given: {batch_img.shape}")

        box = self.get_box(batch_img=batch_img, prev_box=prev_box)
        original_img = torch.permute(batch_img, (0, 1, 4, 2, 3))
        prepared_img = torch.zeros(
            (batch_size, self.num_camera, 3, self.kernel_size, self.kernel_size), device=self.device)
        for i in range(len(box)):
            for j in range(len(box[i])):
                x_center = box[i, j][0]
                y_center = box[i, j][1]
                prepared_img[i, j] = original_img[i, j, :, y_center -
                                                  320:y_center+320, x_center-320:x_center+320]
        return prepared_img, box


class main_object_detector():
    def __init__(self,  img_size=(1920, 1280), kernel_size=640, num_camera=2, engine_path=['models/object-detector/y7_b2.trt', 'models/object-detector/y7_b12.trt'], device=torch.device('cuda:0')):
        self.pred_batch = TensorRTEngine(engine_path=engine_path[1])
        self.pred_single = TensorRTEngine(engine_path=engine_path[0])
        self.auto_segmentation = auto_segmentation()
        self.num_camera = num_camera
        self.device = device
        self.kernel_size = kernel_size
        self.img_size = img_size

    def single_infer(self, batch_img, verbose=True):
        if len(batch_img.shape) == 4:
            batch_img = np.expand_dims(batch_img, axis=0)
        if len(batch_img) > 1:
            raise ValueError(
                "This function only support single batch inference")

        final_det = torch.zeros(self.num_camera*4, device=self.device)
        final_score = torch.zeros(self.num_camera, device=self.device)

        t0 = time.time()
        prepared_img, box = self.auto_segmentation.prepare_imgs(
            batch_img=batch_img, prev_box=[[0, 0], [0, 0]])
        box = box - 320
        t1 = time.time()

        p = self.pred_single.infer(prepared_img[0].ravel())

        t2 = time.time()

        p_class = p[3].reshape(self.num_camera, -1)
        p_box = p[1].reshape(self.num_camera, -1)
        p_score = p[2].reshape(self.num_camera, -1)
        
        for camera in range(0, self.num_camera):
            num_detections = p_class[camera][p_class[camera] == 4].shape[0]

            if not num_detections > 0:
                continue
            else:
                for j in torch.where(p_class[camera] == 4)[0]:
                    det_box = p_box[camera][j*4:j*4+4]
                    det_box[0] += box[0][camera][0]
                    det_box[2] += box[0][camera][0]
                    det_box[1] += box[0][camera][1]
                    det_box[3] += box[0][camera][1]

                    final_score[camera] = p_score[camera, j]
                    final_det[camera*4:camera*4+4] = det_box
                    break

        t3 = time.time()
        if verbose:
            print(
                f"AutoSegmentation : {(t1-t0) * 1000:.2f} ms\nFinal Inference  : {(t2-t1) * 1000:.2f} ms\nPost Processing  : {(t3-t2) * 1000:.2f} ms\nTotal Time       : {(t3-t0) * 1000:.2f} ms")
        return final_det.unsqueeze(0), final_score.unsqueeze(0)

    def all_infer(self, acd, batch_size=6, n_batch_limit=np.inf, desc=""):
        final_batch_det = []
        final_batch_scores = []

        total_batch = min([acd.get_total_batch(
            data_index=i, batch_size=batch_size) for i in range(acd.data_count)])
        for batch_index in tqdm.trange(min(total_batch, n_batch_limit), desc=desc):
            batch_img = acd.get_frame_from_video(
                batch_index, size=(1280, 1920), batch_size=batch_size)
            prepared_img, box = self.auto_segmentation.prepare_imgs(
                batch_img=batch_img, prev_box=[[0, 0], [0, 0]])
            p = self.pred_batch.infer(prepared_img.ravel())
            box = box - 320
            p_box = p[1].reshape(batch_size, 2, -1)
            p_class = p[3].reshape(batch_size, 2, -1)
            p_score = p[2].reshape(batch_size, 2, -1)

            current_batch_det = torch.zeros((batch_size, 4*2))
            current_batch_score = torch.zeros((batch_size, 2))

            for batch in range(batch_size):
                for camera in range(0, 2):
                    num_detections = p_class[batch,
                                             camera][p_class[batch, camera] == 4].shape[0]

                    if not num_detections > 0:
                        continue
                    else:
                        for j in torch.where(p_class[batch, camera] == 4)[0]:
                            det_box = p_box[batch, camera, j*4:j*4+4]
                            det_box[0] += box[batch, camera][0]
                            det_box[2] += box[batch, camera][0]
                            det_box[1] += box[batch, camera][1]
                            det_box[3] += box[batch, camera][1]

                            current_batch_score[batch,
                                                camera] = p_score[batch, camera, j]
                            current_batch_det[batch,
                                              camera*4:camera*4+4] = det_box
                            break
            final_batch_det.append(current_batch_det)
            final_batch_scores.append(current_batch_score)
        final_batch_det = torch.concatenate(final_batch_det)
        final_batch_scores = torch.concatenate(final_batch_scores)
        return final_batch_det, final_batch_scores

    def vis(self, img, boxes, fig=None, ax=None):
        batch_size, num_camera, __, __, __ = img.shape
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(batch_size, num_camera, figsize=(20, 20))
        if len(ax.shape) == 1:
            ax = np.expand_dims(ax, 0)
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
        img = np.ascontiguousarray(img)
        for batch in range(batch_size):
            for camera in range(num_camera):
                temp_img = np.ascontiguousarray(img[batch, camera].copy())
                box = boxes[batch, camera*4:camera*4+4]
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                color = [1, 0, 0]
                cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
                ax[batch, camera].imshow(temp_img)
        return fig, ax
