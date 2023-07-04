from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
import numpy as np
import tqdm
import geopy.distance as gdist
import math
import torch
from types import NoneType
from fastdtw import fastdtw
from torchvision.ops import box_convert


def time_shift_calibration(data=[]):
    if len(data) == 0:
        return None
    elif len(data) == 1:
        return np.expand_dims(np.arange(0, len(data[0])), axis=1)
    else:
        base = data[0]
        others = data[1:]
        start_points = [0]
        for b in others:
            _, calibration = fastdtw(base, b)
            calibration = np.array(calibration)
            delta_calibration = (
                calibration - np.append(calibration[1:], calibration[-1:], axis=0))
            current_base_calibration = delta_calibration[:, 0]
            current_data_calibration = delta_calibration[:, 1]
            start_points.append(np.where(current_data_calibration == -1)
                                [0][0] - np.where(current_base_calibration == -1)[0][0])

        calibration_length = max(
            [start_points[i] + len(data[i]) for i in range(len(data))])
        calibration = np.empty((calibration_length, len(data)))
        calibration[:] = np.nan
        for i in range(len(data)):
            calibration[start_points[i]:start_points[i] +
                        len(data[i]), i] = np.arange(0, len(data[i]))
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

    def get_frame_from_video(self, synced_index_or_batch_n=0, size=None, data_indexes=None, batch_size=None, raw = False):
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
        if raw:
            imgs = np.zeros((batch_size, len(data_indexes), size[0], size[1], 3))
            for data_index in data_indexes:
                start_index = np.where(np.isnan(
                    self.calibration[synced_index:synced_index+batch_size, data_index]) == False)[0]
                if start_index.shape[0] == 0:
                    continue
                self.vidcap[data_index].set(
                    cv2.CAP_PROP_POS_FRAMES, self.calibration[synced_index+start_index[0]][data_index])
                for i in range(start_index[0], batch_size):
                    if synced_index+start_index[0]+i == len(self.calibration) or np.isnan(self.calibration[synced_index+start_index[0]+i][data_index]):
                        break
                    success, image = self.vidcap[data_index].read()
                    if success:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        imgs[i, data_index, :image.shape[0], :image.shape[1]] = image
                    else:
                        break
                return imgs
        else:
            imgs = np.zeros((batch_size, len(data_indexes), size[0], size[1], 3))
            for data_index in data_indexes:
                start_index = np.where(np.isnan(
                    self.calibration[synced_index:synced_index+batch_size, data_index]) == False)[0]
                if start_index.shape[0] == 0:
                    continue
                self.vidcap[data_index].set(
                    cv2.CAP_PROP_POS_FRAMES, self.calibration[synced_index+start_index[0]][data_index])
                for i in range(start_index[0], batch_size):
                    if synced_index+start_index[0]+i == len(self.calibration) or np.isnan(self.calibration[synced_index+start_index[0]+i][data_index]):
                        break
                    success, image = self.vidcap[data_index].read()
                    if success:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        imgs[i, data_index, :image.shape[0], :image.shape[1]] = image
                    else:
                        break
            return torch.from_numpy(imgs).float() / 255
    
    def get_current_dtt(self, synced_index):
        current_calibration = self.calibration[synced_index]
        if np.logical_not(np.isnan(current_calibration)).any():
            # find available aircraft data on the current calibration
            for data_index in range(self.data_count):
                if not np.isnan(current_calibration[data_index]):
                    break    
            coords_2 = (self.aircraft_data[data_index][int(current_calibration[data_index])][0], self.aircraft_data[data_index][int(current_calibration[data_index])][1])
            gt_distance = gdist.geodesic(self.touchdown_target_lat_lon, coords_2).nm
            return gt_distance
        else:
            return -1

    def vis_frame(self, synced_index_or_img, stacked=False, fig=None, ax=None):
        if isinstance(fig, NoneType) or isinstance(ax, NoneType):
            if stacked == True:
                fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            else:
                fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        if isinstance (synced_index_or_img, int):
            synced_index = synced_index_or_img
            if synced_index > len(self.calibration):
                raise "Index out of Range"
            imgs = self.get_frame_from_video(
                synced_index_or_batch_n=synced_index).numpy()
        else:
            imgs = synced_index_or_img
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.cpu().numpy()
            imgs = imgs.astype(str(synced_index_or_img.dtype).split('.')[-1])

        if isinstance(ax, np.ndarray) and len(ax) >= self.data_count:
            ax = ax.reshape(-1,)
            for i in range(self.data_count):

                ax[i].imshow(imgs[0][i])
                ax[i].axis('off')
        else:
            ax.imshow(np.vstack([imgs[0][i] for i in range(self.data_count)]))
            ax.axis('off')
        return fig, ax

    def compute_dtt(self, data_indexes=None):
        if data_indexes == None:
            data_indexes = [i for i in range(self.data_count)]
        all_distances = np.zeros((len(self.calibration), len(data_indexes)))

        for synced_index in tqdm.trange(len(self.calibration), ncols=100):
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
                all_distances[synced_index, i] = (gdist.geodesic((self.touchdown_target_lat_lon[0], self.touchdown_target_lat_lon[1]), coords).nm)
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


def load_compiled_data(ts=[1], ws=[1], rs=[1], unpacked_data = None, convert=True, inference_mode=False, offset=[[0, 0, 0, 0], [0, 0, 0, 0]], divider=[[1, 1, 1, 1], [1, 1, 1, 1]], trajectory_threshold=[np.inf, np.inf], device=torch.device('cpu'), minimum_dist=1, output_directory='output/object_detector/', verbose=True):
    # Only support 2 cameras
    camera_1_data = []
    camera_2_data = []
    
    
    offset = torch.tensor(offset, device=device)
    divider = torch.tensor(divider, device=device)
    
    if unpacked_data == None:
        for t in ts:
            for w in ws:
                for r in rs:
                    with open(f"{output_directory}/t{t}w{w}r{r}.pkl", 'rb') as f:
                        box, scores, gt_distance = pkl.load(f)
                    gt_distance = torch.from_numpy(gt_distance)
                    try:
                        camera_1_det = torch.cat(((box[:, :4].float()[:len(gt_distance)]), gt_distance[:, 0].unsqueeze(
                            1) / 10, scores[:len(gt_distance), 0].unsqueeze(1)), axis=1)
                        camera_2_det = torch.cat(((box[:, 4:].float()[:len(gt_distance)]), gt_distance[:, 0].unsqueeze(
                            1) / 10, scores[:len(gt_distance), 1].unsqueeze(1)), axis=1)
                    except:
                        print(t,w,r)
                        raise ExceptionError

                    # Remove Missing Detection
                    camera_1_data.append(camera_1_det)
                    camera_2_data.append(camera_2_det)
        camera_1_data = torch.cat(camera_1_data)
        camera_2_data = torch.cat(camera_2_data)
    else:
        box, scores, gt_distance = unpacked_data
        if not isinstance(gt_distance, torch.Tensor):
            gt_distance = torch.from_numpy(gt_distance)
        camera_1_data = torch.cat(((box[:, :4].float()[:len(gt_distance)]), gt_distance[:, 0].unsqueeze(
            1) / 10, scores[:, 0].unsqueeze(1)), axis=1)
        camera_2_data = torch.cat(((box[:, 4:].float()[:len(gt_distance)]), gt_distance[:, 0].unsqueeze(
            1) / 10, scores[:, 1].unsqueeze(1)), axis=1)
    
    camera_1_xy = camera_1_data.float().to(device)
    camera_2_xy = camera_2_data.float().to(device)
    
    if convert:
        # Convert to (X_centroid, Y_centroid, X_width, Y_width)
        camera_1_xy = torch.cat((box_convert(camera_1_xy[:, :4], "xyxy", "cxcywh"), camera_1_xy[:, 4:]), dim = 1)
        camera_2_xy = torch.cat((box_convert(camera_2_xy[:, :4], "xyxy", "cxcywh"), camera_2_xy[:, 4:]), dim = 1)

    f_camera_1_available_detection = torch.all(
        torch.logical_not(camera_1_xy[:, :4] == 0), dim=1)
    f_camera_2_available_detection = torch.all(
        torch.logical_not(camera_2_xy[:, :4] == 0), dim=1)

    f_camera_1_gt_more_than_minimum_dist = camera_1_xy[:, -2] > (minimum_dist/10)
    f_camera_2_gt_more_than_minimum_dist = camera_2_xy[:, -2] > (minimum_dist/10)

    # landing trajectory threshold can be determined by plotting Y_centroid against dist)
    f_camera_1_detections_in_landing_trajectory = camera_1_xy[:, 1] < trajectory_threshold[0]
    f_camera_2_detections_in_landing_trajectory = camera_2_xy[:, 1] < trajectory_threshold[1]

    f_camera_single_1 = torch.logical_and(torch.logical_and(
        f_camera_1_available_detection, f_camera_1_gt_more_than_minimum_dist), f_camera_1_detections_in_landing_trajectory)
    f_camera_single_2 = torch.logical_and(torch.logical_and(
        f_camera_2_available_detection, f_camera_2_gt_more_than_minimum_dist), f_camera_2_detections_in_landing_trajectory)

    f_camera_all_available_detection = torch.logical_and(
        f_camera_1_available_detection, f_camera_2_available_detection)
    f_camera_all_gt_more_than_minimum_dist = torch.logical_and(
        f_camera_1_gt_more_than_minimum_dist, f_camera_2_gt_more_than_minimum_dist)

    f_camera_dual = torch.logical_and(torch.logical_and(f_camera_all_available_detection, f_camera_all_gt_more_than_minimum_dist), torch.logical_and(
        f_camera_1_detections_in_landing_trajectory, f_camera_2_detections_in_landing_trajectory))

    camera_1_xy[:, :4] = (camera_1_xy[:, :4] - offset[0]) / divider[0]
    camera_2_xy[:, :4] = (camera_2_xy[:, :4] - offset[1]) / divider[1]
    
    

    if inference_mode:
        camera_1_xy[:, :4][torch.logical_not(
            f_camera_single_1)] = torch.zeros(4, device=device)
        camera_2_xy[:, :4][torch.logical_not(
            f_camera_single_2)] = torch.zeros(4, device=device)

        output_dict = {'x': torch.stack([camera_1_xy[:, :4], camera_2_xy[:, :4]]),
                       'y': torch.stack([camera_1_xy[:, -2:], camera_2_xy[:, -2:]])

                       }
    else:
        output_dict = {'single': {'cam_1': {'x': camera_1_xy[f_camera_single_1][:, :4],
                                            'y': camera_1_xy[f_camera_single_1][:, -2:]},
                                  'cam_2': {'x': camera_2_xy[f_camera_single_2][:, :4],
                                            'y': camera_2_xy[f_camera_single_2][:, -2:]}},
                       'dual': {'cam_1': {'x': camera_1_xy[f_camera_dual][:, :4],
                                          'y': camera_1_xy[f_camera_dual][:, -2:]},
                                'cam_2': {'x': camera_2_xy[f_camera_dual][:, :4],
                                          'y': camera_2_xy[f_camera_dual][:, -2:]}}
                       }
        if verbose:
            try:
                print(f"""Data Statistic:\n
Camera 1:
    Mean    : {torch.mean(camera_1_xy[f_camera_single_1][:, :4], 0).tolist()}
    Std     : {torch.std(camera_1_xy[f_camera_single_1][:, :4], 0).tolist()}
    Min     : {torch.min(camera_1_xy[f_camera_single_1][:, :4], dim = 0).values.tolist()}
    Min-Max : {(torch.max(camera_1_xy[f_camera_single_1][:, :4], dim = 0).values - torch.min(camera_1_xy[f_camera_single_1][:, :4], dim = 0).values).tolist()}""")
            except:
                print("Camera 1 has no available detection")
            try:
                print(f"""
Camera 2:
    Mean    : {torch.mean(camera_2_xy[f_camera_single_2][:, :4], 0).tolist()}
    Std     : {torch.std(camera_2_xy[f_camera_single_2][:, :4], 0).tolist()}
    Min     : {torch.min(camera_2_xy[f_camera_single_2][:, :4], dim = 0).values.tolist()}
    Min-Max : {(torch.max(camera_2_xy[f_camera_single_2][:, :4], dim = 0).values - torch.min(camera_2_xy[f_camera_single_2][:, :4], dim = 0).values).tolist()}
    """)
            except:
                print("Camera 2 has no available detection")
    

    return output_dict


class GeneralDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.gt = y

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        return self.data[idx], self.gt[idx]