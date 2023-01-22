import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import yaml
import tqdm.auto as tqdm
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from utils.dataset import GeneralDataset


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


class inference_ensemble():
    import tqdm as tqdm
    # Results are not meant to be backpropped
    def __init__(self, input_reg_weights_path, sequential_de_weights_path, device):
        self.device = device
        self.input_regs = []
        self.sequential_de = lstm_de().to(self.device)

        if not isinstance(input_reg_weights_path, list):
            input_reg_weights_path = [input_reg_weights_path]

        for i in input_reg_weights_path:
            temp_model = mlp_reg().to(device)
            temp_model.load_state_dict(torch.load(i))
            self.input_regs.append(temp_model)

        self.sequential_de.load_state_dict(
            torch.load(sequential_de_weights_path))

    def __call__(self, x, batch_mode=True):
        if batch_mode == True:
            with torch.no_grad():
                input_reg_out = torch.zeros(
                    (len(x[0]), len(x), 256), device=self.device, requires_grad=False)
                out = torch.zeros(
                    (len(x[0])), device=self.device, requires_grad=False)

                for cam in range(input_reg_out.shape[1]):
                    f = torch.logical_not(torch.all(x[cam] == 0, dim=1))
                    input_reg_out[f, cam] = self.input_regs[cam](x[cam][f])

                for index in tqdm.trange(input_reg_out.shape[0]):
                    ff = torch.any(input_reg_out[index] != 0, dim=1)
                    if ff.any() == True:
                        out[index] = self.sequential_de(
                            input_reg_out[index][ff])[-1].squeeze()
                return out
        else:
            with torch.no_grad():
                input_reg_out = torch.zeros(
                    (len(x[0]), len(x), 256), device=self.device, requires_grad=False)

                for cam in range(len(x)):
                    f = torch.logical_not(torch.all(x[cam] == 0, dim=1))
                    input_reg_out[f, cam] = self.input_regs[cam](x[cam][f])

                ff = torch.any(input_reg_out[index] != 0, dim=1)
                if ff.any() == True:
                    out = self.sequential_de(
                        input_reg_out[index][ff])[-1].squeeze()
                return out

    def plot_error(self, ypred, gt, fig=None, ax=None):
        if fig == None or ax == None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sort_indices = np.argsort(gt.cpu().detach().numpy())
        error = (ypred - gt) * 100 / gt

        ax.set_ylabel('% Error')
        ax.set_xlabel('Airplane Distance (nm)')
        ax.scatter(gt[sort_indices].cpu().detach().numpy() * 10,
                   error[sort_indices].cpu().detach().numpy(), s=2)
        print(f"Mean Absolute % Error : {abs(error).mean()} %")
        print(f"Max % Error           : {abs(error).max()} %")

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
            
    def __init__(self, model_conf, model_state_dict = None, parts_to_load = ['input_reg', 'de', 'aux'], num_camera = 2, device = torch.device('cpu')):
        super(ensemble, self).__init__()

        with open(model_conf, 'rb') as f: 
            conf = yaml.safe_load(f)

        self.device = device
        self.num_camera = num_camera
        self.training_log = {'aux': None, 'de': None}
        self.epochs = {'aux': 0, 'de': 0}
        self.model_state_dict = None
        self.input_regs = nn.ModuleList([nn.Sequential(*self.load_layers(conf['input_reg'])).to(self.device) for cam in range(0, num_camera)])
        self.sequential_de = nn.Sequential(*self.load_layers(conf['de'])).to(self.device)
        self.aux_head =  nn.Sequential(*self.load_layers(conf['aux'])).to(self.device)
        self.transition_size = next(self.aux_head.parameters()).shape[-1]

        if model_state_dict != None:
            self.model_state_dict = torch.load(model_state_dict)
            if self.model_state_dict['num_camera'] == num_camera:
                if 'input_reg' in parts_to_load:
                    self.input_regs.load_state_dict(self.model_state_dict['state_dict']['input_reg'])
                    print("input Reg Loaded")
                if 'aux' in parts_to_load:
                    self.aux_head.load_state_dict(self.model_state_dict['state_dict']['aux'])
                    print("aux Loaded")
                if 'de' in parts_to_load:
                    self.sequential_de.load_state_dict(self.model_state_dict['state_dict']['de'])
                    print("de Loaded")
                self.training_log = self.model_state_dict['training_log']
                self.epochs = self.model_state_dict['epochs']
            else:
                self.model_state_dict = None
         
    
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
    
    def forward(self, x, batch_mode = True):
        if batch_mode == True:
            with torch.no_grad():
                input_reg_out = torch.zeros(
                    (len(x[0]), len(x), self.transition_size), device=self.device, requires_grad=False)
                out = torch.zeros(
                    (len(x[0])), device=self.device, requires_grad=False)

                for cam in range(input_reg_out.shape[1]):
                    f = torch.logical_not(torch.all(x[cam] == 0, dim=1))
                    input_reg_out[f, cam] = self.input_regs[cam](x[cam][f])

                for index in tqdm.trange(input_reg_out.shape[0]):
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
                for cam in range(len(x)):
                    if not torch.any(x[0]==0):
                        input_reg_out[cam] = self.input_regs[cam](x[cam])
                ff = torch.any(input_reg_out != 0, dim=1)
                if ff.any() == True:
                    out = self.sequential_de(
                        input_reg_out[ff].unsqueeze(0)).squeeze()
                return out