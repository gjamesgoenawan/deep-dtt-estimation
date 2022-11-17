from tabulate import tabulate
from fastdtw import fastdtw
import torch
import math
import geopy
import tqdm.notebook as tqdm
import numpy as np
import pickle as pkl
import time
import os
import cv2
import matplotlib.pyplot as plt
print("aircraft-detection custom utils")

def multiway_dtw(data=[]):
    if len(data) == 0:
        return None
    elif len(data) == 1:
        return np.expand_dims(np.arange(0, len(data[0])), axis=1)
    else:
        base = data[0]
        others = data[1:]
        start_ends = []
        for b in others:
            _, calibration = fastdtw(base, b)
            calibration = np.array(calibration)
            start_end = []
            for i in range(0, 2):
                calibration_change = calibration[:, i] - np.concatenate(
                    (calibration[1:, i], [calibration[-1, i]]))
                start = np.where((calibration_change - np.concatenate(
                    ([calibration_change[0]], calibration_change[:-1]))) == -1)[0]
                end = calibration.shape[0] - np.where((calibration_change - np.concatenate(
                    ([calibration_change[0]], calibration_change[:-1]))) == 1)[0][0] - 1
                start = 0 if start.shape[0] == 0 else start[0]
                start_end.append(np.array([start, end]))
            start_ends.append(start_end)
        start_ends = np.array(start_ends)
        max_start_base = start_ends[:, 0, 0].max()
        max_end_base = start_ends[:, 0, 1].max()
        calibration = np.expand_dims(np.concatenate((np.zeros(
            max_start_base, dtype=int) * np.nan, np.arange(0, len(base)), np.zeros(max_end_base, dtype=int) * np.nan)), 1)
        for i in range(len(start_ends)):
            current_calibration = np.expand_dims(np.concatenate((np.zeros(max_start_base - start_ends[i, 0, 0] + start_ends[i, 1, 0], dtype=int) * np.nan, np.arange(
                0, len(others[i])), np.zeros(max_end_base - start_ends[i, 0, 1] + start_ends[i, 1, 1], dtype=int) * np.nan)), 1)
            calibration = np.concatenate(
                (calibration, current_calibration), axis=1)
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

        self.calibration = multiway_dtw(self.aircraft_data)
        self.synced_data_length = len(self.calibration)

    def get_total_batch(self, data_index, batch_size=64):
        return int(math.ceil(self.video_length[data_index] / batch_size))

    def compute_aircraft_to_touchdown_distances(self, data_indexes=None):
        if data_indexes == None:
            data_indexes = [i for i in range(self.data_count)]

        all_distances = []
        for synced_index in tqdm.trange(len(self.calibration)):
            current_calibration = self.calibration[synced_index]
            if np.isnan(current_calibration[data_indexes]).all():
                continue

            # find available aircraft data on the current calibration
            for data_index in data_indexes:
                if not np.isnan(current_calibration[data_index]):
                    break

            coords_2 = (self.aircraft_data[data_index][int(current_calibration[data_index])]
                        [0], self.aircraft_data[data_index][int(current_calibration[data_index])][1])
            all_distances.append(geopy.distance.geodesic(
                (self.touchdown_target_lat_lon[0], self.touchdown_target_lat_lon[1]), coords_2).nm)
        return all_distances

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
                if np.isnan(self.calibration[synced_index+start_index[0]+i][data_index]):
                    break
                success,image = self.vidcap[data_index].read()
                if success:
                    imgs[i, data_index, :image.shape[0], :image.shape[1]] = image
                else:
                    break
        
        return imgs.astype('float32') / 255

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

    # def initiate_yolov7(self):
    #     if self.yolo_model != None:
    #         return None
    #     import yolov7.custom_hubconf as hc
    #     self.yolo_model = hc.custom(path_or_model='yolov7.pt')  # custom example
    #     self.names = self.yolo_model.module.names if hasattr(self.yolo_model, 'module') else self.yolo_model.names

    # def detect_airplanes(self, data_indexes = None, batch_size = 64):
    #     self.initiate_yolov7()
    #     # generating detection using yolov7
    #     if data_indexes == None:
    #         data_indexes = [i for i in range(self.data_count)]

    #     all_detections = []
    #     for data_index in data_indexes:
    #         detections = []
    #         for i in tqdm.trange(0, self.get_total_batch(data_index = data_index, batch_size = batch_size), desc = f"Detecting Objects for data {data_index}"):
    #             pred = self.yolo_model(self.get_frame_from_video_batch(data_index = data_index, index = i, batch_size = batch_size))
    #             detections += pred.pred
    #         all_detections.append(detections)
    #     return all_detections

    # def detect_and_compile(self, data_indexes = None, save_dir = "processed_data", batch_size = 64):
    #     if data_indexes == None:
    #         data_indexes = [i for i in range(self.data_count)]

    #     print("Detecting Airplanes Using Yolov7")
    #     detections = self.detect_airplanes(data_indexes = data_indexes, batch_size = batch_size)
    #     print("Computing Camera to Aircraft Distances")
    #     distances = self.compute_aircraft_to_touchdown_distances(data_indexes = data_indexes)
    #     print("Done!")
    #     filenum = len(os.listdir(save_dir))
    #     if len(data_indexes) == 1:
    #         calibration = np.expand_dims(self.calibration[:, data_indexes], axis = 1)
    #     else:
    #         calibration = self.calibration[:, data_indexes]
    #     with open(os.path.join(save_dir, f'{filenum}.pkl'), 'wb') as f:
    #         pkl.dump([detections, distances, calibration], f)

    # def single_inference(self, synced_index = 0, mono_mode = False, verbose = False, vis = True):
    #     self.initiate_yolov7()
    #     if mono_mode or self.data_count == 1:
    #         if verbose:
    #             print("Inferencing using Mono Camera Model")
    #         self.initiate_mlp()
    #         current_predictor = self.mlp_model
    #         data_indexes = [0]
    #     else:
    #         if verbose:
    #             print("Inferencing using Stereo Camera Model")
    #         self.initiate_mlp_dual()
    #         current_predictor = self.mlp_dual_model
    #         data_indexes = [i for i in range(self.data_count)]

    #     current_frames = self.get_synced_frame_from_video(data_indexes = data_indexes, synced_index = synced_index)
    #     pred = []
    #     torch.cuda.synchronize()
    #     t0 = time.time()
    #     detection = self.yolo_model(current_frames).pred
    #     torch.cuda.synchronize()
    #     t1 = time.time()
    #     for data_index in data_indexes:
    #         if detection[data_index].shape[0] == 0 or int(detection[data_index][0][5]) != 4:
    #             if verbose:
    #                 print(f"No Airplanes Detected in Camera {data_index}")
    #             pred.append(torch.zeros(4).to(self.torch_device))
    #         else:
    #             current_detection = detection[data_index][0]
    #             x_center = (current_detection[2] + current_detection[0]) / 2
    #             y_center =(current_detection[3] + current_detection[1]) / 2
    #             x_span = current_detection[2] - current_detection[0]
    #             y_span = current_detection[3] - current_detection[1]
    #             pred.append(torch.stack((x_center, y_center, x_span, y_span)))
    #     pred = (torch.cat(pred) / current_predictor.scale_pred)
    #     pred_distance = current_predictor(pred).cpu().detach().numpy() * current_predictor.scale_gt
    #     torch.cuda.synchronize()
    #     t2 = time.time()

    #     current_calibration = self.calibration[synced_index]
    #     if not np.isnan(current_calibration[data_indexes]).all():
    #         # find available aircraft data on the current calibration
    #         for data_index in data_indexes:
    #             if not np.isnan(current_calibration[data_index]):
    #                 break
    #         coords_2 = (self.aircraft_data[data_index][int(current_calibration[data_index])][0], self.aircraft_data[data_index][int(current_calibration[data_index])][1])
    #         gt_distance = geopy.distance.geodesic((self.touchdown_target_lat_lon[0], self.touchdown_target_lat_lon[1]), coords_2).nm
    #     else:
    #         gt_distance = np.nan

    #     if verbose:
    #         errors = abs(pred_distance-gt_distance) * 100 / pred_distance
    #         print (tabulate(np.vstack((pred_distance, gt_distance, errors)).transpose(), headers=["Predicted Distance (nm)", "Ground Truth Distance (nm)", "Errors (%)"]))
    #         print(f"YOLO Inference Time : {t1-t0} ({int(1/(t1-t0))} fps)")
    #         print(f"MLP  Inference Time : {t2-t1} ({int(1/(t2-t1))} fps)")
    #         print(f"Cumulative          : {t2-t0} ({int(1/(t2-t0))} fps)")

    #     imgs = []
    #     if vis:
    #         for data_index in range(len(current_frames)):
    #             if detection[data_index].shape[0] == 0:
    #                 imgs.append(current_frames[data_index])
    #             else:
    #                 imgs.append(self.viz_detection(detection = detection[data_index][0], prediction = pred_distance[0], img = current_frames[data_index], gt = gt_distance))
    #     return pred_distance, gt_distance, imgs

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
