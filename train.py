import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
from plot import plot_training_history
import data_preprocess
import my_model
import time

def train_loop(train_loader, test_loader, model, device, criterion, epoch_already, prefix, time_stamp = None, save_path:str = None):
    if save_path:
        if not os.path.isdir(save_path):
            print('Dir does not exist!')
            exit(0)
    best_test_loss = float('inf')
    start_time = time.time()
    if time_stamp == None:
        time_stamp = f'{start_time:.4f}'
    print(f'Training start: {prefix}')
    with open(os.path.join('records', f'{prefix}_timestamp{time_stamp}.txt'), 'a', encoding = 'UTF-8') as f:
        train_losses = list()
        test_losses = list()
        train_errors = list()
        test_errors = list()
        train_errors_abs = list()
        test_errors_abs = list()
        info_list = list()
        mre = my_model.MRE()
        mare = my_model.MARE()
        train_length = len(train_loader.dataset)
        test_length = len(test_loader.dataset)
        epochs = int(input('Please input train epochs: '))
        epoch_per_save = int(input('Please input epochs_per_save: '))
        while(epochs < 1):
            epochs = int(input('Invalid input, num of epochs should be larger than 0: '))
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_error = 0.0
            train_error_abs = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs) # PNN时返回Tensor，Full时返回tuple[Tensor, Tensor]
                loss = criterion(outputs, targets)
                error = mre(outputs, targets)
                error_abs = mare(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 记录每batch的数据
                train_loss += loss.item() * inputs.size(0)
                train_error += error.item() * inputs.size(0)
                train_error_abs += error_abs.item() * inputs.size(0)
            # 计算并记录每个epoch中整个训练集的平均数据
            train_loss = train_loss / train_length
            train_losses.append(train_loss)
            train_error = train_error / train_length
            train_errors.append(train_error)
            train_error_abs = train_error_abs / train_length
            train_errors_abs.append(train_error_abs)
            # 测试模型
            model.eval()
            test_loss = 0.0
            test_error = 0.0
            test_error_abs = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    error = mre(outputs, targets)
                    error_abs = mare(outputs, targets)
                    # 记录每batch的数据
                    test_loss += loss.item() * inputs.size(0)
                    test_error += error.item() * inputs.size(0)
                    test_error_abs += error_abs.item() * inputs.size(0)
            # 计算并记录每个epoch后整个测试集的平均数据
            test_loss = test_loss / test_length
            test_losses.append(test_loss)
            test_error = test_error / test_length
            test_errors.append(test_error)
            test_error_abs = test_error_abs / test_length
            test_errors_abs.append(test_error_abs)
            # 训练记录临时存放在列表中
            info_list.append(f'Epoch {epoch + 1 + epoch_already} | Train Loss: {train_loss:.8f} | Test Loss: {test_loss:.8f} | Train Error: {train_error:.8f} | Test Error: {test_error:.8f} | Train Error_abs: {train_error_abs:.8f} | Test Error_abs: {test_error_abs:.8f}\n')
            if (epoch + 1) % epoch_per_save == 0:
                duration = time.time() - start_time
                print(f'{prefix} | Epoch {epoch + 1 + epoch_already}/{epochs + epoch_already} | Train Loss: {train_loss:.8f} | Test Loss: {test_loss:.8f} | Test Error: {test_error:.8f} | Test Error_abs: {test_error_abs:.8f} | duration: {duration:.2f}s')
                for info in info_list:
                    f.write(info)
                info_list.clear()
                torch.save(model.state_dict(), os.path.join(save_path, f'{prefix}_timestamp{time_stamp}_epoch{epoch + 1 + epoch_already}.pth'))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                model_dict = model.state_dict()
    
    if save_path:
        torch.save(model_dict, os.path.join(save_path, f'best_{prefix}_{best_test_loss}.pth'))
        print(f"Model saved at {os.path.join(save_path, f'best_{prefix}_{best_test_loss}.pth')}")
    print('Training completed!')
    return train_losses, test_losses, train_errors, test_errors, train_errors_abs, test_errors_abs

if __name__ == '__main__':
    save_path = os.path.join(os.getcwd(), 'model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')
    # 选择模型
    model_choice = int(input('Please select a model to train, 1 for PNNreal, 2 for PNNimag, 3 for FullModel: '))
    while(model_choice != 1 and model_choice != 2 and model_choice != 3):
        model_choice = int(input('Invalid input, please input 1, 2 or 3!'))
    if model_choice != 3: # 训练PNN网络
        transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
        #transforms.Normalize(())
        ])
        train_loader_real, val_loader_real, test_loader_real, train_loader_imag, val_loader_imag, test_loader_imag, input_size, output_size, dataname = data_preprocess.prepare_data_PNN(batch_size = 256, transform = transform, subscale = True)
        print('Data Loaded!')
        model = my_model.PNN(input_size = input_size, output_size = output_size, origin = True).to(device)
        print(model)
        criterion = nn.MSELoss()
        lr = 1e-6
        print(f'Learning Rate = {lr}')
        optimizer = optim.Adam(model.parameters(), lr = lr)
        if model_choice == 1:
            prefix = f'PNNreal_{dataname}'
            model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
            if model_dict != None:
                model.load_state_dict(model_dict)
            train_losses, test_losses, train_errors, test_errors, train_errors_abs, test_errors_abs = train_loop(train_loader_real, val_loader_real, model, device, criterion, epoch, prefix, time_stamp, save_path)
            plot_training_history(train_losses, test_losses, train_errors, test_errors, train_errors_abs, test_errors_abs)
        else:
            prefix = f'PNNimag_{dataname}'
            model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
            if model_dict != None:
                model.load_state_dict(model_dict)
            train_losses, test_losses, train_errors, test_errors, train_errors_abs, test_errors_abs = train_loop(train_loader_imag, val_loader_imag, model, device, criterion, epoch, prefix, time_stamp, save_path)
            plot_training_history(train_losses, test_losses, train_errors, test_errors, train_errors_abs, test_errors_abs)
    else: # 训练Fullmodel
        transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
        #transforms.Normalize(())
        ])
        train_loader, val_loader, test_loader, param_size, spectra_size, dataname = data_preprocess.prepare_data_Full(batch_size = 256, transform = transform, subscale = False)
        print('Data Loaded!')

        # real
        PNNreal = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNreal_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNreal_{dataname} ready for training.')
        PNNreal.load_state_dict(model_dict)
        # 禁用 PNNreal 的参数更新
        for param in PNNreal.parameters():
            param.requires_grad = False
        # imag
        PNNimag = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNimag_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNimag_{dataname} ready for training.')
        PNNimag.load_state_dict(model_dict)
        # 禁用 PNNimag 的参数更新
        for param in PNNimag.parameters():
            param.requires_grad = False
        # full
        model = my_model.Fullmodel(input_size = spectra_size, output_size = param_size, PNNreal = PNNreal, PNNimag = PNNimag, origin = False).to(device)
        prefix = f'Fullmodel_{dataname}'
        model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
        if model_dict != None:
            model.load_state_dict(model_dict)
        criterion = my_model.custom_criterion(min_val = 0, max_val = 0, device = device, dataname = dataname, mode = 'mean')
        optimizer = optim.Adam(model.FCblock.parameters(), lr = 1e-5)
        train_losses, test_losses, train_errors, test_errors, train_errors_abs, test_errors_abs = train_loop(train_loader, val_loader, model, device, criterion, epoch, prefix, time_stamp, save_path)
        plot_training_history(train_losses, test_losses, train_errors, test_errors, train_errors_abs, test_errors_abs)