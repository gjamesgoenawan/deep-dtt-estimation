import torch
import tqdm.auto as tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import GeneralDataset
import random
import os
random.seed(0)

try:
    from IPython import display
    notebook_mode = True
except:
    notebook_mode = False


def train_aux(input_regs, aux_head, train_data, test_data, total_epochs=1000, num_camera=2, lr=1e-5, mode='auto', optimizers_algortihm='Adam', training_log=None, optimizers=None, desc='Training Input Regulator'):
    """
#### Function to train `input_regs` with `aux_head`

Input: 
    - `input_regs`           : A list of `input_reg` type models.
    - `aux_head`             : An `aux` type model.
    - `train_data`           : Train Data that can be obtained from `utils.data.load_compiled_data`.
    - `test_data`            : Test  Data that can be obtained from `utils.data.load_compiled_data`.
    - `total_epochs`         : Total Number of epochs to train on.
    - `num_camera`           : Number of camera / views present.
    - `lr`                   : Learning Rate (will be constant throughout the training).
    - `mode`                 : Training Mode (more details below).
    - `optimizers_algortihm` : Optimizer Algorithm (refer to torch.optim).
    - `training_log`         : A list of numpy.ndarray to store errors. If `None`, a new list will be created.
    - `optimizers`           : A list of optimizers that will be used. If `None`, a new list will be created.

Output:
    - `input_regs`           : A list of trained `input_reg` type models.
    - `aux_head`             : A trained `aux` type model.
    - `epochs`               : Total Number of epochs the models are trained on.
    - `training_log`         : A list of numpy.ndarray with previous errors.
    - `optimizers`           : A list of optimizers used.

Note:
- Due to the generally small size of the models, no minibatch is used, hence there are only 1-2 training iterations per epoch.
- Training Modes Available: `'auto'`, `'all'`, `camera n`
    - `all` will train all camera in order.
    - `camera n` will train specific camera. 
    - `auto` will intelligently choose which mode to achieve optimal result
    
  Note that Camera is counted from 1, hence `0` will target `Camera 1`.
  
- It is recommended to pass previous `training_log` and `optimizers` for further training to ensure previous logs are not lost and optimizers state are carried on if necessary.
    """
    global notebook_mode

    if training_log == None:
        mape_train_log = np.empty((0))
        mape_test_log = np.empty((0))
    else:
        mape_train_log = training_log[0]
        mape_test_log = training_log[1]

    if optimizers_algortihm == 'Adam':
        optimizer_alg = torch.optim.Adam
    elif optimizers_algortihm == 'SGD':
        optimizer_alg = torch.optim.SGD

    if optimizers == None or optimizers[0][0].param_groups[-1]['lr'] != lr:
        complete_optimizer = [optimizer_alg(list(model.parameters(
        )) + list(aux_head.parameters()), lr=lr) for model in input_regs]

        neck_optimizer = [optimizer_alg(model.parameters(), lr=lr)
                          for model in input_regs]

    else:
        complete_optimizer = optimizers[0]
        neck_optimizer = optimizers[1]

    if mode == 'auto':
        current_mode = 'all'
    else:
        current_mode = mode

    criterion = torch.nn.MSELoss()
    
    epochs = tqdm.trange(total_epochs, leave=True)

    train_loss = [np.nan] * 2
    test_loss = [np.nan] * 2
    ypred_test = [np.nan] * 2
    ypred_train = [np.nan] * 2
    mape_train = [np.nan] * 2
    mape_test = [np.nan] * 2
    all_training_tracker = 0

    for i in input_regs:
        i.train()
    aux_head.train()

    try:
        for epoch in epochs:
            for i in complete_optimizer:
                i.zero_grad()
            for i in neck_optimizer:
                i.zero_grad()

            if current_mode == 'all':
                for cam in range(num_camera):
                    # for train_x, train_y in singleview_dataloader[cam]:
                    train_x = train_data['single'][f'cam_{cam+1}']['x']
                    train_y = train_data['single'][f'cam_{cam+1}']['y']
                    train_loss[cam] = criterion(aux_head(input_regs[cam](
                        train_x)).squeeze(), train_y[:, 0])
                    train_loss[cam].backward()
                    complete_optimizer[cam].step()
                all_training_tracker += 1

            else:
                cam = current_mode
                train_x = train_data['single'][f'cam_{cam+1}']['x']
                train_y = train_data['single'][f'cam_{cam+1}']['y']
                train_loss[cam] = criterion(aux_head(input_regs[cam](
                    train_x)).squeeze(), train_y[:, 0])
                train_loss[cam].backward()
                neck_optimizer[cam].step()

            if epoch % 500 == 0:
                for i in input_regs:
                    i.eval()
                aux_head.eval()

                for cam in range(num_camera):
                    ypred_test[cam] = aux_head(input_regs[cam](
                        test_data['single'][f'cam_{cam+1}']['x'])).squeeze()
                    ypred_train[cam] = aux_head(input_regs[cam](
                        train_data['single'][f'cam_{cam+1}']['x'])).squeeze()
                    test_loss[cam] = criterion(
                        ypred_test[cam], test_data['single'][f'cam_{cam+1}']['y'][:, 0])
                    mape_test[cam] = ((abs(ypred_test[cam]-test_data['single']
                                           [f'cam_{cam+1}']['y'][:, 0])/test_data['single'][f'cam_{cam+1}']['y'][:, 0]).mean() * 100).item()
                    mape_train[cam] = ((abs(ypred_train[cam]-train_data['single']
                                        [f'cam_{cam+1}']['y'][:, 0])/train_data['single'][f'cam_{cam+1}']['y'][:, 0]).mean() * 100).item()

                display.clear_output(wait=True)
                display.display(epochs.container)
                print(f"""==========================
Epoch {epoch}
Training Mode: {current_mode}
Desc: {desc}

Loss:
    Train:
{''.join([f'        Cam {i + 1} : {train_loss[i]} {os.linesep}' for i in range(len(train_loss))])}
    Test:
{''.join([f'        Cam {i + 1} : {test_loss[i]} {os.linesep}' for i in range(len(test_loss))])}

MAPE
    Train:
{''.join([f'        Cam {i + 1} : {mape_train[i]:.2f} % {os.linesep}' for i in range(len(mape_train))])}
    Test:
{''.join([f'        Cam {i + 1} : {mape_test[i]:.2f} % {os.linesep}' for i in range(len(mape_test))])}
==========================""")

                mape_train_log = np.append(mape_train_log, mape_train.copy())
                mape_test_log = np.append(mape_test_log, mape_test.copy())
                
                if len(mape_train_log) / num_camera > 10 and mode == 'auto' and num_camera > 1:
                    if current_mode == 'all':
                        if np.std(mape_test_log.reshape(-1, num_camera)[-5:], axis=0).mean() < 0.01 and all_training_tracker > 1500:
                            current_mode = mape_test_log[-num_camera:].argmax()

                    else:
                        if mape_test_log[-num_camera:].std() < 0.01:
                            current_mode = 'all'
                            all_training_tracker = 0

                for i in input_regs:
                    i.train()
                aux_head.train()
        return input_regs, aux_head, epoch+1, [complete_optimizer, neck_optimizer], [mape_train_log.reshape(-1, 2), mape_test_log.reshape(-1, 2)]
    except KeyboardInterrupt:
        return input_regs, aux_head, epoch+1, [complete_optimizer, neck_optimizer], [mape_train_log.reshape(-1, 2), mape_test_log.reshape(-1, 2)]


def train_de(input_regs, sequential_de, train_data, test_data, batch_size=32, total_epochs=1000, optimizers_algortihm='Adam', num_camera=2, lr=1e-5, mode='random', training_log=None, optimizers=None, desc = "Training"):
    """
#### Function to train `de`

Input: 
    - `input_regs`           : A list of `input_reg` type models.
    - `sequential_de`        : An `de` type model.
    - `train_data`           : Train Data that can be obtained from `utils.data.load_compiled_data`.
    - `test_data`            : Test  Data that can be obtained from `utils.data.load_compiled_data`.
    - `total_epochs`         : Total Number of epochs to train on.
    - `num_camera`           : Number of camera / views present.
    - `batch_size`           : Size of minibatch.
    - `lr`                   : Learning Rate (will be constant throughout the training).
    - `mode`                 : Training Mode (more details below).
    - `optimizers_algortihm` : Optimizer Algorithm (refer to torch.optim).
    - `training_log`         : A list of numpy.ndarray to store errors. If `None`, a new list will be created.
    - `optimizers`           : A list of optimizers that will be used. If `None`, a new list will be created.

Output:
    - `input_regs`           : A list of trained `input_reg` type models.
    - `aux_head`             : A trained `aux` type model.
    - `epochs`               : Total Number of epochs the models are trained on.
    - `training_log`         : A list of numpy.ndarray with previous errors.
    - `optimizers`           : A list of optimizers used.

Note:
- Since minibatch training is used, number of training iterations varies depending on the `batch_size`.
- Training Modes Available: `'random'`, `camera n`
    - `random` will randomly drop views / camera in the minibatch.
    - `camera n` will constantly drop specific view / camera. 
    
  Note that Camera is counted from 1, hence `0` will target `Camera 1`.
  
- It is recommended to pass previous `training_log` and `optimizers` for further training to ensure previous logs are not lost and optimizers state are carried on if necessary.
    """
    global notebook_mode

    if training_log == None:
        mape_train_log = np.empty((0))
        mape_test_log = np.empty((0))
    else:
        mape_train_log = training_log[0]
        mape_test_log = training_log[1]

    if optimizers_algortihm == 'Adam':
        optimizer_alg = torch.optim.Adam
    elif optimizers_algortihm == 'SGD':
        optimizer_alg = torch.optim.SGD

    if optimizers == None or optimizers[0].param_groups[-1]['lr'] != lr:
        optimizer = optimizer_alg(sequential_de.parameters(), lr=lr)
    else:
        optimizer = optimizers[0]

    criterion = torch.nn.MSELoss()

    epochs = tqdm.trange(total_epochs, leave=True)
    
    depths = ['Shallow :',
              'Deep    :']
    train_loss = np.nan
    test_loss = [np.nan] * 2
    ypred_train = np.nan
    ypred_test = [np.nan] * 2
    mape_train = 0
    mape_test = [np.nan] * 2
    max_dist_test = [np.nan] * 2

    with torch.no_grad():
        if num_camera > 1:
            train_x = torch.stack([input_regs[i](
                train_data['dual'][f'cam_{i+1}']['x']) for i in range(num_camera)], dim=1)
            train_y = torch.stack(
                [train_data['dual'][f'cam_{i+1}']['y'] for i in range(num_camera)], dim=1)

            test_per_cam = [{'x': torch.cat([input_regs[i](test_data['single'][f'cam_{i+1}']['x']) for i in range(num_camera)]).unsqueeze(1),
                             'y': torch.cat([test_data['single'][f'cam_{i+1}']['y'] for i in range(num_camera)]).unsqueeze(1)},
                            {'x': torch.stack([input_regs[i](test_data['dual'][f'cam_{i+1}']['x']) for i in range(num_camera)], dim=1),
                             'y': torch.stack([test_data['dual'][f'cam_{i+1}']['y'] for i in range(num_camera)], dim=1)}]
        else:
            train_x = input_regs[0](train_data['single'][f'cam_1']['x']).unsqueeze(1)
            train_y = train_data['single'][f'cam_1']['y'].unsqueeze(1)
            test_per_cam = [{'x': input_regs[0](test_data['single'][f'cam_1']['x']).unsqueeze(1),
                             'y': test_data['single'][f'cam_1']['y'].unsqueeze(1)}]
            mode = 2
        

    total_data = len(train_x)

    od_dataloader = DataLoader(GeneralDataset(
        train_x, train_y), batch_size=batch_size, shuffle=True)

    sequential_de.train()

    try:
        for epoch in epochs:
            for train_minibatch_x, train_minibatch_y in od_dataloader:
                optimizer.zero_grad()

                if mode == 'random':
                    drop_choice = random.randint(0, 2)
                else:
                    drop_choice = mode
                if drop_choice < int(train_minibatch_x.shape[1]):
                    train_minibatch_x = train_minibatch_x[:,
                                                          drop_choice:drop_choice+1, :]
                else:
                    pass

                ypred = sequential_de(train_minibatch_x)[:, -1].squeeze()
                train_loss = criterion(
                    ypred, train_minibatch_y[:, 0, 0])
                train_loss.backward()
                optimizer.step()

                if epoch % 5 == 0:
                    mape_train += ((abs(ypred-train_minibatch_y[:, 0, 0])/train_minibatch_y[:, 0, 0]).mean(
                    ) * 100).item() * len(train_minibatch_x)

            if epoch % 5 == 0:
                sequential_de.eval()
                cam_loop = 2 if num_camera > 1 else 1
                for cam in range(0, cam_loop):
                    ypred_test[cam] = sequential_de(
                        test_per_cam[cam]['x']).squeeze()
                    test_loss[cam] = criterion(
                        ypred_test[cam], test_per_cam[cam]['y'][:, 0, 0])
                    mape_test[cam] = ((abs(ypred_test[cam]-test_per_cam[cam]['y']
                                      [:, 0, 0])/test_per_cam[cam]['y'][:, 0, 0]).mean() * 100).item()
                    max_dist_test[cam] = abs(
                        ypred_test[cam]-test_per_cam[cam]['y'][:, 0, 0]).max()
                display.clear_output(wait=True)
                display.display(epochs.container)
                print(f"""==========================
Epoch {epoch}
Training Mode: {mode}
Desc: {desc}

Loss:
    Train:
        Current : {train_loss}
        
    Test:
{''.join([f'        {depths[i]} {test_loss[i]} % {os.linesep}' for i in range(len(test_loss))])}
MAPE 
    Train:
        Current : {(mape_train / total_data):.2f}%
    Test:
{''.join([f'        {depths[i]} {mape_test[i]:.2f} % {os.linesep}' for i in range(len(mape_test))])}
    Test (Max):
{''.join([f'        {depths[i]} {max_dist_test[i] * 10:.2f} nm {os.linesep}' for i in range(len(max_dist_test))])}
==========================""")

                mape_train_log = np.append(mape_train_log, mape_train)
                mape_test_log = np.append(mape_test_log, mape_test.copy())
                mape_train = 0
                sequential_de.train()

        return input_regs, sequential_de, epoch+1, [optimizer], [mape_train_log, mape_test_log]

    except KeyboardInterrupt:
        return input_regs, sequential_de, epoch+1, [optimizer], [mape_train_log, mape_test_log]
