import time
import torch
import torch.nn as nn
import os
import shutil
import scipy
import numpy as np
from typing import Tuple, List
class NTNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.W1 = nn.Parameter(torch.Tensor(output_size, input_size, input_size))
        self.W2 = nn.Parameter(torch.Tensor(output_size, input_size, input_size))
        self.V = nn.Parameter(torch.Tensor(output_size, input_size * 2))
        self.b = nn.Parameter(torch.Tensor(1, output_size))

        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        nn.init.normal_(self.V)
        nn.init.normal_(self.b)

    def forward(self, x):
        xTW1x = torch.einsum('bd, kdj, bj -> bk', x, self.W1, x) # (batch, d).T * (out, d, d) * (batch, d) -> (batch, out)
        xx = x * x
        xTW2xx = torch.einsum('bd, kdj, bj -> bk', x, self.W2, xx) # (batch, d).T * (out, d, d) * (batch, d) -> (batch, out)
        x_v = torch.cat([x, xx], dim = 1)# (batch, d * 2)
        Vx = torch.einsum('bd, kd -> bk', x_v, self.V)# (batch, d * 2) * (out, d * 2) -> (batch, out)
        return xTW1x + xTW2xx + Vx + self.b

class PNN(nn.Module):
    def __init__(self, input_size, output_size, origin = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        dropout_rate = 0.3
        if origin:
            ntnout_size = 50
            self.FCblock = nn.Sequential(
                NTNLayer(input_size, ntnout_size),
                nn.ReLU(),
                nn.Linear(ntnout_size, 500),
                nn.ReLU(),
                nn.Dropout(p = dropout_rate),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Dropout(p = dropout_rate),
                nn.Linear(500, 200),
                nn.ReLU(),
                nn.Dropout(p = dropout_rate),
                nn.Linear(200, output_size),
            )
        elif origin == False:
            ntnout_size = 128
            self.FCblock = nn.Sequential(
                NTNLayer(input_size, ntnout_size),
                nn.Tanh(),
                nn.Linear(ntnout_size, 1024),
                nn.ReLU(),
                nn.Dropout(p = dropout_rate),
                nn.Linear(1024, 4096),
                nn.ReLU(),
                nn.Dropout(p = dropout_rate),
                nn.Linear(4096, 1024),
                nn.ReLU(),
                nn.Dropout(p = dropout_rate),
                nn.Linear(1024, 301),
            )
    def forward(self, x):
        x = self.FCblock(x)
        return x

class Fullmodel(nn.Module):
    def __init__(self, input_size, output_size, PNNreal, PNNimag, origin = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if origin:
            self.FCblock = nn.Sequential(
                nn.Linear(input_size, 500),
                nn.Sigmoid(),
                nn.Linear(500, 500),
                nn.Sigmoid(),
                nn.Linear(500, 500),
                nn.Sigmoid(),
                nn.Linear(500, 50),
                nn.Sigmoid(),
                nn.Linear(50, output_size)
            )
        else:
            self.FCblock = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Linear(32, output_size)
            )
        self.PNNreal = PNNreal
        self.PNNimag = PNNimag

    def forward(self ,x):
        x = self.FCblock(x)
        real = self.PNNreal(x)
        imag = self.PNNimag(x)
        spectrum = torch.sqrt(real * real + imag * imag)
        return x, spectrum

class MRE(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, pre, true):
        if isinstance(pre, tuple):
            pre = pre[1]
        relative_error = (pre - true) / (true + self.eps)
        return torch.mean(relative_error)

class MARE(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, pre, true):
        if isinstance(pre, tuple):
            pre = pre[1]
        relative_error = (pre - true) / (true + self.eps)
        absolute_relative_error = torch.abs(relative_error)
        return torch.mean(absolute_relative_error)

class custom_criterion(nn.Module):
    def __init__(self, min_val, max_val, device, dataname = None, mode = 'mean'):
        super().__init__()
        if dataname != None:
            magnitude_data = np.genfromtxt(os.path.join('data', 'magnitude_data_' + dataname + '.csv'), delimiter=',', skip_header=1)
            parameter = magnitude_data[:, :2].astype(np.float32)
            self.min_vals = torch.tensor(np.min(parameter) / 100, dtype = torch.float32).to(device)
            self.max_vals = torch.tensor(np.max(parameter) / 100, dtype = torch.float32).to(device)
        else:
            self.min_vals = torch.tensor(min_val, dtype = torch.float32).to(device)
            self.max_vals = torch.tensor(max_val, dtype = torch.float32).to(device)
        print(f'Params limit: [{self.min_vals, self.max_vals}]')
        self.mode = mode
    def forward(self, pre, true):
        params, spectrum = pre
        #spec_range = torch.max(true) - torch.min(true) + 1e-3
        #mse_loss = torch.mean((spectrum - true) ** 2) / spec_range
        mse_loss = torch.mean((spectrum - true) ** 2)
        params_clipped = torch.min(torch.max(params, self.min_vals), self.max_vals)
        if self.mode == 'mean':
            p_loss = torch.mean((params - params_clipped) ** 2)
        elif self.mode == 'sum':
            p_loss = torch.sum((params - params_clipped) ** 2)
        return mse_loss + p_loss
       
def get_model_dict(prefix:str, device) -> Tuple[dict, str, int]:
    filepath = os.path.join(os.getcwd(), 'model')
    file_list = os.listdir(filepath)
    model_list = list()
    final_list = list()
    epoch_dict = dict()
    print(f'Searching models in {filepath}.')
    for filename in file_list:
        if filename.endswith('.pth'):
            if filename.find(prefix) != -1:
                model_list.append(filename)
                epoch_index = filename.find('epoch')
                if epoch_index == -1:
                    continue
                epoch_index_end = filename.find('.pth')
                epoch = int(filename[epoch_index + 5 : epoch_index_end])
                time_index = filename.find('timestamp')
                time_index_end = filename.find('_epoch')
                time_stamp = filename[time_index + 9 : time_index_end]
                if not time_stamp in epoch_dict.keys():
                    epoch_dict[time_stamp] = epoch
                elif epoch > epoch_dict[time_stamp]:
                    epoch_dict[time_stamp] = epoch
    for filename in model_list:
        epoch_index = filename.find('epoch')
        if epoch_index == -1:
            final_list.append(filename)
            continue
        epoch_index_end = filename.find('.pth')
        epoch = int(filename[epoch_index + 5 : epoch_index_end])
        time_index = filename.find('timestamp')
        time_index_end = filename.find('_epoch')
        time_stamp = filename[time_index + 9 : time_index_end]
        if epoch == epoch_dict[time_stamp]:
            final_list.append(filename)

    for i in range(len(final_list)):
        print(f'{i + 1}. {final_list[i]}')
    if len(final_list) == 0:
        choice = -1
    else:
        choice = int(input('Please choose a model to load or input 0 for a new model: ')) - 1
    if choice == -1:
        return None, None, 0
    else:
        epoch_index = final_list[choice].find('epoch')
        if epoch_index == -1:
            model_dict = torch.load(os.path.join(filepath, final_list[choice]), map_location=torch.device(device), weights_only = True)
            return model_dict, None, None
        epoch_index_end = final_list[choice].find('.pth')
        epoch = int(final_list[choice][epoch_index + 5 : epoch_index_end])
        time_index = final_list[choice].find('timestamp')
        time_index_end = final_list[choice].find('_epoch')
        time_stamp = final_list[choice][time_index + 9 : time_index_end]
        print(f'Loading model: {final_list[choice]}.')
        model_dict = torch.load(os.path.join(filepath, final_list[choice]), map_location=torch.device(device)) # , weights_only = True
        return model_dict, time_stamp, epoch
    
#def get_model_records(prefix:str)
if __name__ == '__main__':
    while True:
        filepath = os.path.join(os.getcwd(), 'model')
        file_list = os.listdir(filepath)
        model_list = list()
        final_list = list()
        epoch_dict = dict()
        print(f'Searching models in {filepath}.')
        for filename in file_list:
            if filename.endswith('.pth'):
                model_list.append(filename)
                epoch_index = filename.find('epoch')
                if epoch_index == -1:
                    continue
                epoch_index_end = filename.find('.pth')
                epoch = int(filename[epoch_index + 5 : epoch_index_end])
                time_index = filename.find('timestamp')
                time_index_end = filename.find('_epoch')
                time_stamp = filename[time_index + 9 : time_index_end]
                if not time_stamp in epoch_dict.keys():
                    epoch_dict[time_stamp] = epoch
                elif epoch > epoch_dict[time_stamp]:
                    epoch_dict[time_stamp] = epoch
        for filename in model_list:
            epoch_index = filename.find('epoch')
            if epoch_index == -1:
                final_list.append(filename)
                continue
            epoch_index_end = filename.find('.pth')
            epoch = int(filename[epoch_index + 5 : epoch_index_end])
            time_index = filename.find('timestamp')
            time_index_end = filename.find('_epoch')
            time_stamp = filename[time_index + 9 : time_index_end]
            if epoch == epoch_dict[time_stamp]:
                final_list.append(filename)
        print("""1. Choose a model to delete.\n2. Clean models.(Delete models which don't start with "best" or have the max epoch of a timestamp.)""")
        choice = int(input('Your choice (0 for exit): '))
        if choice == 0:
            exit(0)
        if choice == 1:
            for i in range(len(final_list)):
                print(f'{i + 1}. {final_list[i]}')
            choice = int(input('Your choice (0 for exit): '))
            if choice == 0:
                exit(0)
            if final_list[choice - 1].startswith('best'):
                shutil.move(os.path.join('model', final_list[choice - 1]), 'trash')
            else:
                time_index = final_list[choice - 1].find('timestamp')
                time_index_end = final_list[choice - 1].find('_epoch')
                time_stamp = final_list[choice - 1][time_index + 9 : time_index_end]
                for filename in model_list:
                    if filename.find(time_stamp) != -1:
                        shutil.move(os.path.join('model', filename), 'trash')
                if int(input('Delete record or not, 1 for yes, 0 for no: ')) == 1:
                        filepath = os.path.join(os.getcwd(), 'records')
                        file_list = os.listdir(filepath)
                        for filename in file_list:
                            if filename.find(time_stamp) != -1:
                                shutil.move(os.path.join('records', filename), 'trash')
        elif choice == 2:
            for timestamp in epoch_dict.keys():
                for filename in model_list:
                    if filename.find(timestamp) != -1:
                        if filename.find(str(epoch_dict[time_stamp])) == -1:
                            shutil.move(os.path.join('model', filename), 'trash')
