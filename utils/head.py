import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import yaml
import tqdm.auto as tqdm
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from utils.dataset import GeneralDataset
import torch.nn.init as init
import utils.object_detector as od
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
    def __init__(self, model_conf, offset = None, divider = None, model_state_dict = None, num_camera = 2, img_size=(1920, 1280), kernel_size=640, engine_path=['models/object-detector/y7_b2.trt', 'models/object-detector/y7_b12.trt'], device = torch.device('cuda:0')):
        self.device = device
        self.input_regs = []
        self.num_camera = num_camera
        self.offset = torch.tensor(offset if offset != None else ([[0,0,0,0]] * self.num_camera), device = device)
        self.divider = torch.tensor(divider if divider != None else ([[1,1,1,1]] * self.num_camera), device = device)
        self.ensemble = ensemble(model_conf=model_conf, model_state_dict=model_state_dict, parts_to_load=['input_reg', 'de', 'aux'], num_camera=self.num_camera, device=self.device)
        self.mod = od.main_object_detector(img_size=img_size, kernel_size=kernel_size, num_camera=self.num_camera, engine_path=engine_path)

    def single_infer(self, batch_img, verbose=True):
        if batch_img.device != self.device:
            batch_img = batch_img.to(self.device)
        torch.cuda.synchronize()
        t0 = time.time()
        box, score = self.mod.single_infer(batch_img, verbose=False)
        t1 = time.time()
        box = box.reshape(-1, 4)
        
        formated_box = torch.zeros(box.shape, device=self.device).float()
        formated_box[:, 0] = (box[:, 0] + box[:, 2]) / 2
        formated_box[:, 1] = (box[:, 1] + box[:, 3]) / 2
        formated_box[:, 2] = (box[:, 2] - box[:, 0])
        formated_box[:, 3] = (box[:, 3] - box[:, 1])
        for i in range(self.num_camera):
            if (formated_box[i] == 0).all() == False:
                formated_box[i] = (formated_box[i] - self.offset[i]) / self.divider[i]
        t2 = time.time()
        pred = self.ensemble(formated_box.unsqueeze(1), batch_mode = False)
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
            
            if isinstance(pred, torch.Tensor) and isinstance(gt, torch.Tensor):
                label_pred = f'PRED : {pred:.2f}'
                label_gt =   f'GT   : {gt:.2f}'

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
        self.epochs = {'aux': 0, 'de': 0}
        self.model_state_dict = None
        self.input_regs = nn.ModuleList([nn.Sequential(*self.load_layers(conf['input_reg'])).to(self.device) for cam in range(0, num_camera)])
        self.sequential_de = nn.Sequential(*self.load_layers(conf['de'])).to(self.device)
        self.aux_head =  nn.Sequential(*self.load_layers(conf['aux'])).to(self.device)
        self.transition_size = next(self.aux_head.parameters()).shape[-1]

        if initializer == 'xavier':
            self.apply(weight_init)

        if model_state_dict != None:
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

    def forward(self, x, batch_mode = True, verbose = None):
        if verbose == None:
            verbose = self.verbose

        if batch_mode == True:
            with torch.no_grad():
                input_reg_out = torch.zeros(
                    (len(x[0]), len(x), self.transition_size), device=self.device, requires_grad=False)
                out = torch.zeros(
                    (len(x[0])), device=self.device, requires_grad=False)

                for cam in range(input_reg_out.shape[1]):
                    f = torch.logical_not(torch.all(x[cam] == 0, dim=1))
                    input_reg_out[f, cam] = self.input_regs[cam](x[cam][f])
                
                if verbose:
                    iterable = tqdm.trange(input_reg_out.shape[0])
                else:
                    iterable = range(input_reg_out.shape[0])
                for index in iterable:
                    ff = torch.any(input_reg_out[index] != 0, dim=1)
                    if ff.any() == True:
                        if len(input_reg_out[index][ff].shape) == 2:
                            current_x = input_reg_out[index][ff].unsqueeze(0)
                        else:
                            current_x = input_reg_out[index][ff]
                        out[index] = self.sequential_de(
                            current_x)[-1].squeeze()
                return out
        else:
            with torch.no_grad():
                input_reg_out = torch.zeros(
                    (len(x), self.transition_size), device=self.device, requires_grad=False)
                for cam in range(input_reg_out.shape[0]):
                    if not (x[cam] == 0).all():
                        input_reg_out[cam] = self.input_regs[cam](x[cam][0])

                ff = torch.any(input_reg_out != 0, dim=1)
                out = torch.nan
                if ff.any() == True:
                    out = self.sequential_de(
                        input_reg_out[ff].unsqueeze(0)).squeeze()
                return out
    
    def eval_data(self, train_data, test_data, num_camera = 2, batch_size = 32):
        criterion = torch.nn.MSELoss()
        train_loss_aux = [None] * num_camera
        test_loss_aux = [None] * num_camera
        ypred_test_aux = [None] * num_camera
        ypred_train_aux = [None] * num_camera
        mape_train_aux = [None] * num_camera
        mape_test_aux = [None] * num_camera
        train_loss_de = None
        test_loss_de = [None] * num_camera
        ypred_test_de = [None] * num_camera
        mape_train_de = 0
        mape_test_de = [None] * num_camera
        max_dist_test_de = [None] * num_camera
        

        for i in self.input_regs:
            i.eval()
        self.aux_head.eval()
        self.sequential_de.eval()
        with torch.no_grad():
            for cam in range(num_camera):
                # for train_x, train_y in singleview_dataloader[cam]:
                train_x = train_data['single'][f'cam_{cam+1}']['x']
                train_y = train_data['single'][f'cam_{cam+1}']['y']
                train_loss_aux[cam] = criterion(self.aux_head(self.input_regs[cam](
                    train_x)).squeeze(), train_y[:, 0])

                ypred_test_aux[cam] = self.aux_head(self.input_regs[cam](
                    test_data['single'][f'cam_{cam+1}']['x'])).squeeze()
                ypred_train_aux[cam] = self.aux_head(self.input_regs[cam](
                    train_data['single'][f'cam_{cam+1}']['x'])).squeeze()
                test_loss_aux[cam] = criterion(
                    ypred_test_aux[cam], test_data['single'][f'cam_{cam+1}']['y'][:, 0])
                mape_test_aux[cam] = ((abs(ypred_test_aux[cam]-test_data['single']
                                [f'cam_{cam+1}']['y'][:, 0])/test_data['single']
                                [f'cam_{cam+1}']['y'][:, 0]).mean() * 100).item()
                mape_train_aux[cam] = ((abs(ypred_train_aux[cam]-train_data['single']
                                    [f'cam_{cam+1}']['y'][:, 0])/train_data['single']
                                    [f'cam_{cam+1}']['y'][:, 0]).mean() * 100).item()

                train_x = torch.stack([self.input_regs[i](
                    train_data['dual'][f'cam_{i+1}']['x']) for i in range(num_camera)], dim=1)
                train_y = torch.stack(
                    [train_data['dual'][f'cam_{i+1}']['y'] for i in range(num_camera)], dim=1)

                test_per_cam = [{'x': torch.stack([self.input_regs[i](test_data['dual'][f'cam_{i+1}']['x']) for i in range(num_camera)], dim=1),
                                'y': torch.stack([test_data['dual'][f'cam_{i+1}']['y'] for i in range(num_camera)], dim=1)},
                                {'x': torch.cat([self.input_regs[i](test_data['single'][f'cam_{i+1}']['x']) for i in range(num_camera)]).unsqueeze(1),
                                'y': torch.cat([test_data['single'][f'cam_{i+1}']['y'] for i in range(num_camera)]).unsqueeze(1)}]

            od_dataloader = DataLoader(GeneralDataset(
                train_x, train_y), batch_size=batch_size, shuffle=True)

            total_data = len(train_x)

            for train_minibatch_x, train_minibatch_y in od_dataloader:
                ypred = self.sequential_de(train_minibatch_x)[:, -1].squeeze()
                train_loss_de = criterion(ypred, train_minibatch_y[:, 0, 0])
                mape_train_de += ((abs(ypred-train_minibatch_y[:, 0, 0])/train_minibatch_y[:, 0, 0]).mean() * 100).item() * len(train_minibatch_x)

            for cam in range(0, 2):
                ypred_test_de[cam] = self.sequential_de(test_per_cam[cam]['x']).squeeze()
                test_loss_de[cam] = criterion(ypred_test_de[cam], test_per_cam[cam]['y'][:, 0, 0])
                mape_test_de[cam] = ((abs(ypred_test_de[cam]-test_per_cam[cam]['y'][:, 0, 0])/test_per_cam[cam]['y'][:, 0, 0]).mean() * 100).item()
                max_dist_test_de[cam] = abs(ypred_test_de[cam]-test_per_cam[cam]['y'][:, 0, 0]).max()


        print(f"""==========================
MODEL EVALUATION

Trained Epochs : 
    Aux : {self.epochs['aux']}
    De  : {self.epochs['de']}

Input Regularization
Loss:
    Train:
        Cam 1 : {train_loss_aux[0]}
        Cam 2 : {train_loss_aux[1]}
    Test:
        Cam 1 : {test_loss_aux[0]}
        Cam 2 : {test_loss_aux[1]}

MAPE
    Train:
        Cam 1 : {mape_train_aux[0]:.2f}%
        Cam 2 : {mape_train_aux[1]:.2f}%
    Test:
        Cam 1 : {mape_test_aux[0]:.2f}%
        Cam 2 : {mape_test_aux[1]:.2f}%
    
Distance Estimator
Loss:
    Train:
        Current : {train_loss_de}

    Test:
        Deep    : {test_loss_de[0]}
        Shallow : {test_loss_de[1]}

MAPE 
    Train:
        Current : {(mape_train_de / total_data):.2f}%
    Test:
        Deep    : {mape_test_de[0]:.2f}%
        Shallow : {mape_test_de[1]:.2f}%
    Test (Max):
        Deep    : {max_dist_test_de[0]:.2f} nm
        Shallow : {max_dist_test_de[1]:.2f} nm
==========================""")
    


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
