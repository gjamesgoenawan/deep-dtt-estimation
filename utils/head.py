import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import yaml
import tqdm.auto as tqdm
import numpy as np
from types import NoneType
from utils.object_detector import example_association_function
from torch.utils.data import DataLoader
from utils.dataset import GeneralDataset
import torch.nn.init as init
import utils.object_detector as od
from torchvision.ops import box_convert
import time
import cv2


class lstm_de(nn.Module):
    def __init__(self):
        super(lstm_de, self).__init__()
        self.LSTM = nn.LSTM(input_size=256, hidden_size=256,
                            num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 1),
                                nn.ReLU())

    def forward(self, x):
        # for i in x:
        out, (cs, hs) = self.LSTM(x)
        out = self.fc(out)
        return out


class auxiliary_head(nn.Module):
    def __init__(self):
        super(auxiliary_head, self).__init__()
        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        return out


class mlp_reg(nn.Module):
    def __init__(self):
        super(mlp_reg, self).__init__()
        self.fc_1 = nn.Linear(4, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc_1(x)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        out = self.relu(out)
        return out


class mlp_dual(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_gt = 8
        self.scale_pred = 100
        self.fc = nn.Sequential(nn.Linear(8, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                nn.ReLU())

    def forward(self, x):
        return self.fc(x)


class end2end():
    def __init__(self, model_conf, offset = None, divider = None, model_state_dict = None, num_camera = 2, img_size=(1920, 1280), kernel_size=640, engine_path=['models/object-detector/y7_b1.trt', 'models/object-detector/y7_b12.trt'], association_function = example_association_function, device = torch.device('cuda:0')):
        self.device = device
        self.input_regs = []
        self.num_camera = num_camera
        self.association_function = association_function
        self.offset = torch.tensor(offset if offset != None else ([[0,0,0,0]] * self.num_camera), device = device)
        self.divider = torch.tensor(divider if divider != None else ([[1,1,1,1]] * self.num_camera), device = device)
        self.ensemble = ensemble(model_conf=model_conf, model_state_dict=model_state_dict, parts_to_load=['input_reg', 'de', 'aux'], num_camera=self.num_camera, device=self.device)
        self.mod = od.main_object_detector(img_size=img_size, kernel_size=kernel_size, num_camera=self.num_camera, engine_path=engine_path)

    def single_infer(self, batch_img, verbose=True):
        if batch_img.device != self.device:
            batch_img = batch_img.to(self.device)
        torch.cuda.synchronize()
        t0 = time.time()
        box, score = self.mod.infer(batch_img, verbose=False, association_function = self.association_function)
        t1 = time.time()
        
        current_input = box[0].reshape(int(box[0].shape[1]/4), -1, 4)
        current_input = box_convert(current_input, 'xyxy', 'cxcywh')
        current_input[0] = (current_input[0] - self.offset[0]) / self.divider[0]
        current_input[1] = (current_input[1] - self.offset[1]) / self.divider[1]
        
        t2 = time.time()
        pred = self.ensemble(current_input, verbose = False)[0]
        t3 = time.time()
        if verbose:
            print(f"""Time Profile:

Object Detector    : {((t1-t0) * 1000):.2f} ms
Preprocessing      : {((t2-t1) * 1000):.2f} ms
Distance Estimator : {((t3-t2) * 1000):.2f} ms

Total Time         : {((t3-t0) * 1000):.2f} ms ({1/(t3-t0)} FPS)
            """)
        return box, score, pred.item() * 10

    def vis(self, img, box, score=None, pred=None, gt=None, size =(1080, 1920), fig = None, ax = None):
        img = img[0]
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1,img.shape[0],figsize=(20,10))
        ax = ax.ravel()
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
        else:
            img = np.ascontiguousarray(img)
            
        for i in range(len(box)):
            c1, c2 = ((int(box[i][0]), int(box[i][1])), (int(box[i][2]), int(box[i][3])))
            if c1 == (0,0) or c2 == (0,0):
                continue
            
            tl = 1
            tf = 2
            color = [1,0,0]
            cv2.rectangle(img[i], c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
            
            if not (isinstance(pred, NoneType) or isinstance(gt, NoneType)):
                label_pred = f'PRED : {pred:.2f}'
                label_gt =   f'GT    : {gt:.2f}'

                t_size = cv2.getTextSize(label_pred, 0, tl, tf)[0]

                if c2[0] + 20 + t_size[0] < img.shape[1]:
                    cv2.putText(img[i], label_pred, (c2[0] + 20, c1[1]), 0, tl , [1, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img[i], label_gt, (c2[0] + 20, c1[1] + 40), 0, tl , [0, 0, 1], thickness=tf, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(img[i], label_pred, (c1[0] - t_size[0] - 20, c1[1]), 0, tl , [1, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
                    cv2.putText(img[i], label_gt, (c1[0] - t_size[0] - 20, c1[1] + 40), 0, tl , [0, 0, 1], thickness=tf, lineType=cv2.LINE_AA)

        for i in range(len(ax)):
            ax[i].imshow(img[i][:size[0], :size[1]])
        
        return fig, ax

class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor[: , -1]
    
class ensemble(nn.Module):
    def load_layers(self, layer_list):
        result = []
        for layer in layer_list:
            if layer[0] == 'Linear':
                for i in range(len(layer[1])-1):
                    result.append(nn.Linear(layer[1][i], layer[1][i+1]))
                    result.append(nn.LeakyReLU())
            if layer[0] == 'LSTM':
                result.append(nn.LSTM(input_size=layer[1][0], hidden_size=layer[1][1],num_layers=layer[2], batch_first=True))
                result.append(extract_tensor())
        return result
            
    def __init__(self, model_conf, model_state_dict = None, parts_to_load = ['input_reg', 'de', 'aux'], num_camera = 2, initializer = 'kaiming', device = torch.device('cpu'), verbose = "False"):
        super(ensemble, self).__init__()

        with open(model_conf, 'rb') as f: 
            conf = yaml.safe_load(f)

        self.verbose = verbose
        self.device = device
        self.num_camera = num_camera
        self.training_log = {'aux': None, 'de': None}
        self.empty_result = torch.tensor([-1], device = self.device)
        self.epochs = {'aux': 0, 'de': 0}
        self.model_state_dict = None
        self.input_regs = nn.ModuleList([nn.Sequential(*self.load_layers(conf['input_reg'])).to(self.device) for cam in range(0, num_camera)])
        self.sequential_de = nn.Sequential(*self.load_layers(conf['de'])).to(self.device)
        self.aux_head =  nn.Sequential(*self.load_layers(conf['aux'])).to(self.device)
        self.transition_size = next(self.aux_head.parameters()).shape[-1]

        if initializer == 'xavier':
            self.apply(weight_init)

        if model_state_dict is not None:
            self.model_state_dict = torch.load(model_state_dict)
            if self.model_state_dict['num_camera'] == num_camera:
                if 'input_reg' in parts_to_load:
                    self.input_regs.load_state_dict(self.model_state_dict['state_dict']['input_reg'])
                    if verbose: print("Input_Reg", end=" ")
                if 'aux' in parts_to_load:
                    self.aux_head.load_state_dict(self.model_state_dict['state_dict']['aux'])
                    if verbose: print("Aux", end=" ")
                if 'de' in parts_to_load:
                    self.sequential_de.load_state_dict(self.model_state_dict['state_dict']['de'])
                    if verbose: print("DE", end=" ")
                self.training_log = self.model_state_dict['training_log']
                self.epochs = self.model_state_dict['epochs']
                if verbose: print("Loaded")
            else:
                self.model_state_dict = None

    def forward(self, x, verbose = True):
        if verbose:
            iterable = tqdm.trange(x.shape[1])
        else:
            iterable = range(x.shape[1])
        output = []
        with torch.no_grad():
            for batch in iterable:
                current_output = []
                for cam in range(self.num_camera):
                    current_input = x[cam][batch]
                    if current_input.sum() != 0:
                        current_output.append(self.input_regs[cam](current_input))
                if len(current_output) >= 1:
                    output.append(self.sequential_de(torch.stack(current_output, axis = 0).unsqueeze(0)).squeeze(1))
                else: 
                    output.append(self.empty_result)
            return torch.cat(output)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
