import torch.nn as nn
import torch

class LSTM_DE(nn.Module):
    def __init__(self):
        super(LSTM_DE, self).__init__()
        self.LSTM = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 2, batch_first = True)
        self.fc = nn.Sequential(nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 1),
                                nn.ReLU()
                               )
        
    def forward(self, x):
        #for i in x:
        out, (cs, hs) = self.LSTM(x)
        out = self.fc(out)
        return out
        
class auxiliary_head(nn.Module):
    def __init__(self):
        super(auxiliary_head, self).__init__()
        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128,1)
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
        self.fc_3 = nn.Linear(128,256)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc_1(x)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        out = self.relu(out)
        return out

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_gt = 8
        self.scale_pred = 100
        self.fc = nn.Sequential(nn.Linear(4,128),
                                nn.ReLU(),
                                nn.Linear(128, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 1),
                                nn.ReLU())
    def forward(self, x):
        return self.fc(x)

class mlp_dual(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_gt = 8
        self.scale_pred = 100
        self.fc = nn.Sequential(nn.Linear(8,256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                nn.ReLU())
    def forward(self, x):
        return self.fc(x)

class mlp_dual_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_gt = 8
        self.scale_pred = 100
        self.mask = torch.tensor([[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[0,0,0,0,1,1,1,1],[1,1,1,1,0,0,0,0]]).to('cuda:0')
        self.fc = nn.Sequential(nn.Linear(8,256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1),
                                nn.ReLU())
    def forward(self, x):
        if self.training == True:
            return self.fc(self.mask[torch.randint(low = 0, high = 8, size = [1]).item()] * x)
        else:
            return self.fc(x)
