import tensorrt as trt
import numpy as np
import time
import cv2
import torch
import math
import matplotlib.pyplot as plt
import tqdm
from types import NoneType
from torchvision.ops import box_iou
from collections import OrderedDict, namedtuple

def example_association_function(final_box, final_score):
    empty_box = torch.zeros((1,4), device = final_box[0][0].device)
    empty_score = torch.zeros((1, 1), device = final_score[0][0].device)
    associated_box = []
    associated_score = []
    for batch in range(len(final_box)):
        current_batch_box = []
        current_batch_score = []
        for cam in range(len(final_box[batch])):
            ff = [final_box[batch][cam][:, 1] < 910]
            temp_box = final_box[batch][cam][ff][:1]
            temp_score = final_score[batch][cam][ff][:1]
            
            if len(temp_box) == 0:
                current_batch_box.append(empty_box)
                current_batch_score.append(empty_score)
            else: 
                current_batch_box.append(temp_box)
                current_batch_score.append(temp_score)
        associated_box.append(torch.cat(current_batch_box, dim = 1))
        associated_score.append(torch.stack(current_batch_score, dim = 1))
    return associated_box, associated_score

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
            name = self.model.get_tensor_name(index)
            dtype = trt.nptype(self.model.get_tensor_dtype(self.model[index]))
            shape = tuple(self.model.get_tensor_shape(self.model[index]))
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
    
def torch_delete(tensors, indices):
    mask = torch.ones(tensors[0].shape[0], dtype=torch.bool)
    mask[indices] = False
    return [i[mask] for i in tensors]

def merge_box_and_score(box1, box2, score1, score2):
    f_less = (box1 > box2)[:2]
    f_more = (box1 < box2)[2:]
    temp = box1.clone()
    temp[:2][f_less] = box2[:2][f_less]     
    temp[2:][f_more] = box2[2:][f_more]
    return temp, max(score1, score2)

def should_merge(box1, box2, close_dist = 10, tolerance = 15):
    # one side of a dimension is less than close_dist and other dimension fully covered.
    #      ________
    #     |        |                 _____
    #     |        |                |     |
    #     |        | <-close_dist-> |_____|
    #     |________|
    #
    if  ((box1[0] - box2[2]) < close_dist or (box1[2] - box2[0]) < close_dist) and ((box1[1] - tolerance  <= box2[1] and box1[3] + tolerance >= box2[3]) or (box1[1] + tolerance >= box2[1] and box1[3] - tolerance <= box2[3])) \
        or ((box1[1] - box2[3]) < close_dist or (box1[3] - box2[1]) < close_dist) and ((box1[0] - tolerance <= box2[0] and box1[2] + tolerance >= box2[2]) or (box1[0] + tolerance >= box2[0] and box1[2] - tolerance <= box2[2])):
        f_less = (box1 > box2)[:2]
        f_more = (box1 < box2)[2:]
        temp = box1.clone()
        temp[:2][f_less] = box2[:2][f_less]     
        temp[2:][f_more] = box2[2:][f_more]
        return True, temp
    else:
        return False, None

def is_intersect(boxes, reference_box):
    # Check if the boxes are intersecting with a reference box.
    return torch.logical_and(torch.max(boxes[:, 1], reference_box[1]) < torch.min(boxes[:, 3], reference_box[3]), torch.max(boxes[:, 0], reference_box[0]) < torch.min(boxes[:, 2], reference_box[2]))

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
        batch, camera_count, channel, __, __ = batch_img.shape
        return torch.permute(batch_img.reshape(batch, camera_count, channel, self.n_tile_v, 640, self.n_tile_h, 640), (0,1,3,5,2,4,6)).ravel().reshape(batch, -1)

    def get_aoi(self, batch_img, prev_box=None):
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

        if batch_img.shape[1:] != (self.num_camera, 3, self.resize_height, self.resize_width):
            raise ValueError(
                f"Images must be in the shape of (batch, {self.num_camera}, 3, {self.resize_height}, {self.resize_width}). Given: {batch_img.shape}")

        split_imgs = self.split_image(batch_img)

        batch_aoi = []
        for batch in range(n_batch):
            p = self.pred_batch.infer(split_imgs[batch])
            p_count = p[0].reshape(self.num_camera, -1)
            p_box = p[1].reshape(self.num_camera, -1, p[1].shape[1], p[1].shape[2])
            p_class = p[3].reshape(self.num_camera, -1, p[1].shape[1])
            current_batch_aoi = []
            for cam in range(0, self.num_camera):
                current_cam_boxes = torch.zeros((0, 4), device = self.device)
                for i in range(0, len(p_box[cam])):
                    temp_boxes = p_box[cam][i][:p_count[cam][i]][p_class[cam][i][:p_count[cam][i]] == 4] + self.origins[i]
                    for box1 in temp_boxes:
                        if len(current_cam_boxes) == 0:
                            current_cam_boxes = temp_boxes
                        already_merged = False
                        for index, box2 in enumerate(current_cam_boxes):
                            merge, temp_box = should_merge(box1,box2)
                            if merge:
                                current_cam_boxes[index] = temp_box
                                already_merged = True
                        if not already_merged:
                            current_cam_boxes = torch.cat((current_cam_boxes, box1.unsqueeze(0)))
                current_batch_aoi.append(current_cam_boxes)
            batch_aoi.append(current_batch_aoi)
        return batch_aoi

    def prepare_imgs(self, batch_img, prev_box=None):
        if len(batch_img.shape) == 4:
            batch_img = np.expand_dims(batch_img, axis=0)

        num_camera = batch_img.shape[1]

        if num_camera != self.num_camera:
            raise ValueError(
                f"Number of Camera is invalid. Required = {self.num_camera}, Provided = {num_camera}.\n Note that images provided must be in the shape of (batch, {self.num_camera}, {self.resize_height}, {self.resize_width}, 3)")

        if not (batch_img.shape[1:] == (self.num_camera, 3, self.resize_height, self.resize_width) or batch_img.shape[1:] == (self.num_camera, self.resize_height, self.resize_width, 3)):
            raise ValueError(
                f"Images must be in the shape of (batch, {self.num_camera}, 3, {self.resize_height}, {self.resize_width}). Given: {batch_img.shape}")

        # Boxes indicates Area Of Interest (AOI)
        if batch_img.shape[2] == self.resize_height:
            source_img = torch.permute(batch_img, (0, 1, 4, 2, 3))
        else:
            source_img = batch_img
        batch_aoi = self.get_aoi(batch_img=source_img, prev_box=prev_box)
        
        batch_prepared_images = []
        batch_prepared_origins = []
        for batch in range(len(batch_aoi)):
            current_batch_images = []
            current_batch_origins = []
            for cam in range(len(batch_aoi[batch])):
                current_cam_images = torch.zeros((batch_aoi[batch][cam].shape[0], 3, self.kernel_size, self.kernel_size), device=self.device)
                current_cam_origins = []
                current_cam_boxes = batch_aoi[batch][cam].int()
                for index, box in enumerate(current_cam_boxes):
                    # scale 1
                    if (box[2] - box[0] <= self.kernel_size) and (box[3] - box[1] <= self.kernel_size):
                        box_half_dim = int(self.kernel_size/2)
                        scale = 1

                        center = ((box[2]+box[0])/2).int().item(), ((box[3]+box[1])/2).int().item()
                        source_x_start = 0 if center[0]-box_half_dim <0 else center[0]-box_half_dim
                        source_y_start = 0 if center[1]-box_half_dim <0 else center[1]-box_half_dim
                        temp_image = source_img[batch, cam, :, source_y_start:center[1]+box_half_dim, source_x_start:center[0]+box_half_dim]
                        canvas_x_start = box_half_dim-center[0] if center[0]-box_half_dim<0 else 0
                        canvas_x_end = self.img_size[0] - center[0]+box_half_dim if center[0]+box_half_dim>self.img_size[0] else self.kernel_size
                        canvas_y_start = box_half_dim-center[1] if center[1]-box_half_dim<0 else 0
                        canvas_y_end = self.img_size[1] - center[1]+box_half_dim if center[1]+box_half_dim>self.img_size[1] else self.kernel_size
                        current_cam_images[index, :, canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = temp_image
                        current_cam_origins.append([center[0]-box_half_dim, center[1]-box_half_dim, scale])

                     # scale 2
                    elif (box[2] - box[0] <= self.kernel_size * 2) and (box[3] - box[1] <= self.kernel_size * 2):
                        box_half_dim = (self.kernel_size)
                        scale = 2
                        
                        center = ((box[2]+box[0])/2).int().item(), ((box[3]+box[1])/2).int().item()
                        source_x_start = 0 if center[0]-box_half_dim <0 else center[0]-box_half_dim
                        source_y_start = 0 if center[1]-box_half_dim <0 else center[1]-box_half_dim
                        temp_image = source_img[batch, cam, :, source_y_start:center[1]+box_half_dim:scale, source_x_start:center[0]+box_half_dim:scale]
                        canvas_x_start = int((box_half_dim-center[0])/scale) if center[0]-box_half_dim<0 else 0
                        canvas_x_end = int((self.img_size[0] - center[0]+box_half_dim)/scale) if center[0]+box_half_dim>self.img_size[0] else self.kernel_size
                        canvas_y_start = int((box_half_dim-center[1])/scale) if center[1]-box_half_dim<0 else 0
                        canvas_y_end = int((self.img_size[1] - center[1]+box_half_dim)/scale) if center[1]+box_half_dim>self.img_size[1] else self.kernel_size
                        current_cam_images[index, :, canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = temp_image
                        current_cam_origins.append([center[0]-box_half_dim, center[1]-box_half_dim, scale])
                current_batch_origins.append(torch.tensor(current_cam_origins, device=self.device))
                current_batch_images.append(current_cam_images)
            batch_prepared_origins.append(current_batch_origins)
            batch_prepared_images.append(current_batch_images)
        return batch_prepared_images, batch_prepared_origins, batch_aoi

    def vis(self, prepared_imgs):
        for batch in prepared_imgs:
            temp_img = [i.cpu().numpy().transpose(0,2,3,1) for i in batch]
            for i in temp_img:
                fig, ax = plt.subplots(1, len(i))
                if len(i) == 1:
                    ax.imshow(i[0])
                else:
                    for index, j in enumerate(i):
                        ax[index].imshow(j)

class main_object_detector():
    def __init__(self,  img_size=(1920, 1280), kernel_size=640, num_camera=2, iou_threshold=0, engine_path=['models/object-detector/y7_b1.trt', 'models/object-detector/y7_b12.trt'], device=torch.device('cuda:0')):
        single_engine_path, batch_engine_path = engine_path
        self.pred_single = TensorRTEngine(engine_path=single_engine_path)
        self.auto_segmentation = auto_segmentation(img_size=img_size, kernel_size=kernel_size, num_camera=num_camera, engine_path=batch_engine_path, device=torch.device('cuda:0'))
        self.num_camera = num_camera
        self.device = device
        self.kernel_size = kernel_size
        self.img_size = img_size
        self.iou_threshold=iou_threshold

    def infer(self, batch_img, association_function = None, verbose=True):
        if len(batch_img.shape) == 4:
            batch_img = np.expand_dims(batch_img, axis=0)

        t0 = time.time()
        prepared_imgs, prepared_origins, prepared_aoi = self.auto_segmentation.prepare_imgs(batch_img=batch_img)
        t1 = time.time()

        final_box = []
        final_score = []
        for batch in range(len(prepared_imgs)):
            current_batch_box = []
            current_batch_score = []
            for cam in range(len(prepared_imgs[batch])):
                current_cam_box = []
                current_cam_score = []
                for index, current_img in enumerate(prepared_imgs[batch][cam]):
                    p_count, p_box, p_score, p_class = self.pred_single.infer(current_img)
                    origin_x, origin_y, scale = prepared_origins[batch][cam][index]
                    current_box = p_box[0, :p_count[0]][p_class[0, :p_count[0]] == 4] * scale
                    current_box[:, ::2] += origin_x
                    current_box[:, 1::2] += origin_y
                    if len(current_box) > 0:
                        ff = is_intersect(current_box, prepared_aoi[batch][cam][index])
                        current_cam_box.append(current_box[ff])
                        current_cam_score.append(p_score[0, :p_count[0]][p_class[0, :p_count[0]] == 4][ff])
                if len(current_cam_box) == 0 and len(current_cam_score) == 0:
                    current_batch_box.append(torch.zeros((1, 4), device = self.device))
                    current_batch_score.append(torch.zeros((1), device = self.device))
                else:
                    current_cam_box = torch.cat(current_cam_box)
                    current_cam_score = torch.cat(current_cam_score)

                    while True:
                        ious = box_iou(current_cam_box, current_cam_box).fill_diagonal_(0)
                        likely_same = torch.where(ious > self.iou_threshold)
                        if len(likely_same[0]) == 0:
                            break
                        else:
                            current_cam_box[likely_same[0][0]], current_cam_score[likely_same[0][0]]  = merge_box_and_score(current_cam_box[likely_same[0][0]], current_cam_box[likely_same[1][likely_same[0][0]]], current_cam_score[likely_same[0][0]], current_cam_score[likely_same[1][likely_same[0][0]]])
                            current_cam_box, current_cam_score = torch_delete([current_cam_box, current_cam_score], likely_same[1][likely_same[0][0]])
    
                    current_batch_box.append(current_cam_box)
                    current_batch_score.append(current_cam_score)

            final_box.append(current_batch_box)
            final_score.append(current_batch_score)
        t2 = time.time()

        if not isinstance(association_function, NoneType):
            final_box, final_score = association_function(final_box, final_score)

        t3 = time.time()

        if verbose:
            print(
                f"AutoSegmentation : {(t1-t0) * 1000:.2f} ms\nFinal Inference  : {(t2-t1) * 1000:.2f} ms\nPost-Porcessing  : {(t3-t2) * 1000:.2f} ms\nTotal Time       : {(t3-t0) * 1000:.2f} ms")
        return final_box, final_score

    def infer_acd(self, acd, association_function, batch_size=8, n_batch_limit=np.inf, desc=""):
        final_box = []
        final_score = []
        total_batch = min([acd.get_total_batch(
            data_index=i, batch_size=batch_size) for i in range(acd.data_count)])
        for batch_index in tqdm.trange(min(total_batch, n_batch_limit), desc=desc, ncols=100):
            batch_img = acd.get_frame_from_video(
                batch_index, size=(1280, 1920), batch_size=8)
            current_batch_final_box, current_batch_final_score = self.infer(batch_img, association_function = association_function, verbose = False)
            final_box += current_batch_final_box
            final_score += current_batch_final_score
        return final_box, final_score

    def vis(self, img, boxes=None, fig=None, ax=None):
        batch_size, num_camera, __, __, __ = img.shape
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(batch_size, num_camera, figsize=(20, 20))
        if batch_size==1:
            ax = np.expand_dims(ax, 0)
        if num_camera==1:
            ax = np.expand_dims(ax, 1)
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
        img = np.ascontiguousarray(img)
        
        for batch in range(batch_size):
            for camera in range(num_camera):
                temp_img = np.ascontiguousarray(img[batch, camera].copy())
                if not isinstance(boxes, NoneType):
                    for index, box in enumerate(boxes[batch]):
                        curr_box = box[camera*4:camera*4+4]
                        x0 = int(curr_box[0])
                        y0 = int(curr_box[1])
                        x1 = int(curr_box[2])
                        y1 = int(curr_box[3])
                        color = [1, 0, 0]
                        cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
                ax[batch, camera].imshow(temp_img)
        return fig, ax
