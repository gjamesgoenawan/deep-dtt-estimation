import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import tqdm as tqdm
import numpy as np


class LSTM_DE(nn.Module):
    def __init__(self):
        super(LSTM_DE, self).__init__()
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


class input_reg(nn.Module):
    def __init__(self):
        super(input_reg, self).__init__()
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


class ensemble():
    # Results are not meant to be backpropped
    def __init__(self, input_reg_weights_path, sequential_de_weights_path, device):
        self.device = device
        self.input_regs = []
        self.sequential_de = LSTM_DE().to(self.device)

        if not isinstance(input_reg_weights_path, list):
            input_reg_weights_path = [input_reg_weights_path]

        for i in input_reg_weights_path:
            temp_model = input_reg().to(device)
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
