from tabulate import tabulate
from fastdtw import fastdtw
import torch
import math
import geopy.distance as gdist
import tqdm.notebook as tqdm
import numpy as np
import pickle as pkl
import time
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
print("aircraft-detection custom utils")

def time_shift_calibration(data = []):
    if len(data) == 0:
        return None
    elif len(data) == 1:
        return np.expand_dims(np.arange(0,len(data[0])),axis = 1)
    else:
        base = data[0]
        others = data[1:]
        start_points = [0]
        for b in others:
            _, calibration = fastdtw(base,b)
            calibration = np.array(calibration)
            delta_calibration = (calibration - np.append(calibration[1:], calibration[-1:], axis = 0))
            current_base_calibration = delta_calibration[:, 0]
            current_data_calibration = delta_calibration[:, 1]
            start_points.append(np.where(current_data_calibration == -1)[0][0] - np.where(current_base_calibration == -1)[0][0])
        
        calibration_length = max([start_points[i] + len(data[i]) for i in range(len(data))])
        calibration = np.empty((calibration_length, len(data)))
        calibration[:] = np.nan
        for i in range(len(data)):
            calibration[start_points[i]:start_points[i]+len(data[i]), i] = np.arange(0, len(data[i]))
        return calibration

class aircraft_camera_data():
    def __init__(self, data_sources=[], touchdown_target_lat_lon=(0, 0)):
        if not torch.cuda.is_available():
            raise ("CUDA device not found!")
        self.aircraft_data = []
        self.vidcap = []
        self.video_length = []
        self.video_width = []
        self.video_height = []
        self.yolo_model = None
        self.mlp_model = None
        self.mlp_dual_model = None
        self.torch_device = 'cuda:0'
        self.data_count = len(data_sources)
        self.touchdown_target_lat_lon = touchdown_target_lat_lon
        for data_index in range(self.data_count):
            if data_sources[data_index][0] != None:
                with open(data_sources[data_index][0], 'rb') as f:
                    self.aircraft_data.append(np.array(pkl.load(f)))
            if data_sources[data_index][1] != None:
                self.vidcap.append(cv2.VideoCapture(
                    data_sources[data_index][1]))
                self.video_length.append(
                    int(self.vidcap[data_index].get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                self.video_width.append(
                    int(self.vidcap[data_index].get(cv2.CAP_PROP_FRAME_WIDTH)))
                self.video_height.append(
                    int(self.vidcap[data_index].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.calibration = time_shift_calibration(self.aircraft_data)
        self.synced_data_length = len(self.calibration)

    def get_total_batch(self, data_index, batch_size=64):
        return int(math.ceil(self.video_length[data_index] / batch_size))

    def get_frame_from_video(self, synced_index_or_batch_n=0, size = None, data_indexes=None, batch_size = None):
        if size == None:
            size = (max(self.video_height), max(self.video_width))
        if batch_size != None: 
            synced_index = synced_index_or_batch_n * batch_size
        else:
            synced_index = synced_index_or_batch_n
            batch_size = 1
        if synced_index >= self.synced_data_length:
            raise Exception(
                f"Index given is out of range {self.synced_data_length}")
        if data_indexes == None:
            data_indexes = [i for i in range(self.data_count)]
        imgs = np.zeros((batch_size, len(data_indexes), size[0], size[1], 3))
        for data_index in data_indexes:
            start_index = np.where(np.isnan(self.calibration[synced_index:synced_index+batch_size, data_index]) == False)[0]
            if start_index.shape[0] == 0:
                continue
            self.vidcap[data_index].set(cv2.CAP_PROP_POS_FRAMES, self.calibration[synced_index+start_index[0]][data_index])
            for i in range(start_index[0], batch_size):
                if synced_index+start_index[0]+i == len(self.calibration) or np.isnan(self.calibration[synced_index+start_index[0]+i][data_index]):
                    break
                success,image = self.vidcap[data_index].read()
                if success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    imgs[i, data_index, :image.shape[0], :image.shape[1]] = image
                else:
                    break
                
        return torch.from_numpy(imgs).float() / 255

    def viz_data(self, synced_index, fig=None, ax=None):
        if fig == None or ax == None :
            fig, ax = plt.subplots(1,2, figsize=(20,20))
        if synced_index > len(self.calibration):
            raise "Index out of Range"
        imgs = self.get_frame_from_video(synced_index_or_batch_n=synced_index)
        if isinstance(ax, np.ndarray) and len(ax) >= self.data_count:
            ax = ax.reshape(-1,)
            for i in range(self.data_count):
                ax[i].imshow(imgs[0][i][:, :, ::-1])
                ax[i].axis('off')
        else:
            ax.imshow(np.vstack([imgs[0][i] for i in range(self.data_count)])[:, :, ::-1])
            ax.axis('off')
        return fig, ax
    
    def compute_dtt(self, data_indexes=None):
        if data_indexes == None:
            data_indexes = [i for i in range(self.data_count)]
        all_distances = np.zeros((len(self.calibration), len(data_indexes)))

        for synced_index in tqdm.trange(len(self.calibration)):
            current_calibration = self.calibration[synced_index]
            if np.isnan(current_calibration[data_indexes]).all():
                continue
            # find available aircraft data on the current calibration
            for i in range(len(data_indexes)):
                data_index = data_indexes[i]
                if np.isnan(current_calibration[data_index]):
                    continue
                coords = (self.aircraft_data[data_index][int(current_calibration[data_index])]
                            [0], self.aircraft_data[data_index][int(current_calibration[data_index])][1])
                all_distances[synced_index, i] = (gdist.geodesic(
                    (self.touchdown_target_lat_lon[0], self.touchdown_target_lat_lon[1]), coords).nm)
        return all_distances

    # def viz_detection(self, detection, prediction, img, gt = None):
    #     if gt != None:
    #         label_gt = f'GT   : {gt:.2f} nm'
    #     label_pred = f'Pred : {prediction:.2f} nm'
    #     img = np.ascontiguousarray(img)
    #     if detection.shape[0] == 0:
    #         return img
    #     c1, c2 = (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3]))
    #     tl = 1
    #     tf = 2
    #     color = [255,0,0]
    #     cv2.rectangle(img, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
    #     t_size = cv2.getTextSize(label_pred, 0, tl, tf)[0]
    #     if c2[0] + 20 + t_size[0] < img.shape[1]:
    #         cv2.putText(img, label_pred, (c2[0] + 20, c1[1]), 0, tl , [225, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    #         cv2.putText(img, label_gt, (c2[0] + 20, c1[1] + 40), 0, tl , [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
    #     else:
    #         cv2.putText(img, label_pred, (c1[0] - t_size[0] - 20, c1[1]), 0, tl , [225, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    #         cv2.putText(img, label_gt, (c1[0] - t_size[0] - 20, c1[1] + 40), 0, tl , [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
    #     return img


def convert_to_model_format(detections, distances, calibration, scale_pred=100, scale_gt=8, skip_missing=True):
    data_count = len(detections)
    pred = []
    gt = []
    for i in range(len(calibration)):
        if np.isnan(calibration[i]).any():
            continue
        skip = False
        scene_pred = []
        for j in range(len(detections)):
            if detections[j][int(calibration[i][j])].shape[0] == 0 or detections[j][int(calibration[i][j])][0][5] != 4:
                skip = True
                scene_pred.append(np.array([0, 0, 0, 0]))
            else:
                temp_detections = detections[j][int(
                    calibration[i][j])][0].cpu().detach().numpy()
                x_center = (temp_detections[2] + temp_detections[0]) / 2
                y_center = (temp_detections[3] + temp_detections[1]) / 2
                x_span = temp_detections[2] - temp_detections[0]
                y_span = temp_detections[3] - temp_detections[1]
                scene_pred.append(
                    np.array([x_center, y_center, x_span, y_span]))
        if skip == True and skip_missing:
            continue
        pred.append(np.concatenate(scene_pred))
        gt.append(distances[i])
    pred = (np.vstack(pred) / scale_pred)
    gt = (np.expand_dims(np.array(gt), axis=1) / scale_gt)
    return pred, gt


def load_compiled_data(ts=[1], ws=[1], rs=[1], offset = [[0,0,0,0],[0,0,0,0]], divider = [[1,1,1,1],[1,1,1,1]], device = torch.device('cpu')):
    camera_1_dets = []
    camera_2_dets = []
    camera_1_ys = []
    camera_2_ys = []

    offset = torch.tensor(offset, device =device)
    divider = torch.tensor(divider, device =device)

    for t in ts:
        for w in ws:
            for r in rs:
                with open(f"output/object_detector/t{t}w{w}r{r}.pkl", 'rb') as f:
                        box, scores, gt_distance = pkl.load(f)
                gt_distance = torch.from_numpy(gt_distance)
                camera_1_det = torch.cat(((box[:, :4].float()[:len(gt_distance)]), gt_distance[:, 0].unsqueeze(1) / 10, scores[:, 0].unsqueeze(1)), axis = 1)
                camera_2_det = torch.cat(((box[:, 4:].float()[:len(gt_distance)]), gt_distance[:, 0].unsqueeze(1) / 10, scores[:, 1].unsqueeze(1)), axis = 1)
                
                # Remove Missing Detection
                
                f_camera_1_available_detection = torch.all(torch.logical_not(camera_1_det[:, :4] == 0), dim = 1)
                f_camera_2_available_detection = torch.all(torch.logical_not(camera_2_det[:, :4] == 0), dim = 1)
                f_camera_1_gt_more_than_1nm = camera_1_det[:, -2] > 0.1
                f_camera_2_gt_more_than_1nm = camera_2_det[:, -2] > 0.1
                
                f_camera_1 = torch.logical_and(f_camera_1_available_detection, f_camera_1_gt_more_than_1nm)
                f_camera_2 = torch.logical_and(f_camera_2_available_detection, f_camera_2_gt_more_than_1nm)
                
                
                camera_1_det = camera_1_det[f_camera_1].float()
                camera_2_det = camera_2_det[f_camera_2].float()
    

                camera_1_dets.append(camera_1_det[:, :4])
                camera_1_ys.append(camera_1_det[:, -2:])
                camera_2_dets.append(camera_2_det[:, :4])
                camera_2_ys.append(camera_2_det[:, -2:])
                
    camera_1_y = torch.cat(camera_1_ys).to(device)
    camera_2_y = torch.cat(camera_2_ys).to(device)      
    camera_1_dets = torch.cat(camera_1_dets)
    camera_2_dets = torch.cat(camera_2_dets)
    
    # Convert to (X_centroid, Y_centroid, X_width, Y_width)
    camera_1_x = torch.zeros(camera_1_dets[:, :4].shape, device = device)
    camera_2_x = torch.zeros(camera_2_dets[:, :4].shape, device = device)
    camera_1_x[:, 0] = (camera_1_dets[:, 0] + camera_1_dets[:, 2]) / 2
    camera_1_x[:, 1] = (camera_1_dets[:, 1] + camera_1_dets[:, 3]) / 2
    camera_1_x[:, 2] = (camera_1_dets[:, 2] - camera_1_dets[:, 0])
    camera_1_x[:, 3] = (camera_1_dets[:, 3] - camera_1_dets[:, 1])
    camera_2_x[:, 0] = (camera_2_dets[:, 0] + camera_2_dets[:, 2]) / 2
    camera_2_x[:, 1] = (camera_2_dets[:, 1] + camera_2_dets[:, 3]) / 2
    camera_2_x[:, 2] = (camera_2_dets[:, 2] - camera_2_dets[:, 0])
    camera_2_x[:, 3] = (camera_2_dets[:, 3] - camera_2_dets[:, 1])
    
    camera_1_x = (camera_1_x - offset[0]) / divider[0]
    camera_2_x = (camera_2_x - offset[1]) / divider[1]
    
    print(f"""Data Statistic:
          Camera 1:
          Mean: {torch.mean(camera_1_x, 0).tolist()}
          Std : {torch.std(camera_1_x, 0).tolist()}
          
          Camera 2:
          Mean: {torch.mean(camera_2_x, 0).tolist()}
          Std : {torch.std(camera_2_x, 0).tolist()}
          """)
    return camera_1_x, camera_1_y, camera_2_x, camera_2_y

class ObjectDetectorDataset(Dataset):
    def __init__(self, x, y):
        print(x.shape, y.shape)
        self.data = x
        self.gt = y

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        return self.data[idx], self.gt[idx]