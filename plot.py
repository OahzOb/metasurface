#############
import matplotlib
matplotlib.use('Agg')
# windows10系统中需要指定'Agg'才能使用plot_sample函数，否则会内存溢出OOM（具体取决于数据集和可用内存大小，这并不是一个因为死循环而导致的OOM）。
# 具体原因是每次新建fig都会占用新内存，而plt.close(fig)时不会释放出所有占用的内存，导致每保存一张图片都会占用新的内存（debug时大概几MB/张的额外无法释放的内存）。
# 指定'Agg'后端，因为'Agg'后端有更激进的内存管理策略（deepseek说的）
# linux系统中不会遇到这个问题，linux中matplotlib默认是'Agg'后端（deepseek说的，怪不得linux中plt.show()没法展示图片）。
# 由此推理，windows10系统中matplotlib使用的后端多占用的内存是为图片展示服务，尽管我的代码中没有展示图片的部分，而linux中'Agg'后端直接不提供图片展示功能，所以减少了内存占用。
#############
import matplotlib.pyplot as plt
import time
import numpy as np
import data_preprocess
from torchvision import transforms
import torch
import torch.nn as nn
import my_model
import os
import csv
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import re
from torch.utils.data import Dataset, DataLoader, random_split, Subset

def plot_parameter_distribution():
    parameter, _, phase, param_size, spectra_size, dataname = data_preprocess.read_file_to_numpy(subscale=False, origin=True)
    '''
    绘制parameter的二维分布图，每个元素是[长, 宽]，范围在[150, 650]之间
    
    :param parameter: 包含长和宽的二维数组，形状为(n_samples, 2)
    '''
    # 提取长和宽
    lengths = parameter[:, 0] * 100  # 因为函数中返回的是除以100后的值
    widths = parameter[:, 1] * 100
    
    # 创建画布
    plt.figure(figsize=(10, 8))
    
    # 绘制二维直方图
    plt.hist2d(lengths, widths, bins=20, range=[[150, 650], [150, 650]], cmap='viridis')
    plt.colorbar(label='Number of Samples')
    
    # 设置坐标轴和标题
    plt.xlabel('Length (mm)')
    plt.ylabel('Width (mm)')
    plt.title('Parameter Distribution (Length vs Width)')
    
    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

def plot_training_history(train_losses:list, test_losses:list, train_errors:list, test_errors:list, train_errors_abs:list, test_errors_abs:list) -> None:
    """
    绘制损失曲线与误差曲线。
    :params:    [train_losses]: 训练损失列表。
                [test_losses]: 测试损失列表。
                [train_errors]: 训练误差列表。
                [test_errors]: 测试误差列表。
                [train_errors_abs]: 训练绝对误差列表。
                [test_errors_abs]: 测试绝对误差列表。

    :return:    None
    """
    epochs = range(1, len(train_losses) + 1)
    _, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 9))
    # 绘制完整损失曲线
    axs[0, 0].plot(epochs, train_losses, label = 'Train Loss')
    axs[0, 0].plot(epochs, test_losses, label = 'Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    # 绘制最后的10%epochs的曲线
    index = int(0.9 * len(train_losses))
    axs[0, 1].plot(epochs[index:], train_losses[index:], label = 'Train Loss')
    axs[0, 1].plot(epochs[index:], test_losses[index:], label = 'Validation Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    # 绘制误差曲线
    axs[1, 0].plot(epochs, train_errors, label = 'Train Error')
    axs[1, 0].plot(epochs, test_errors, label = 'Validation Error')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Error')
    axs[1, 0].legend()
    # 绘制绝对误差曲线
    axs[1, 1].plot(epochs, train_errors_abs, label = 'Train Error_abs')
    axs[1, 1].plot(epochs, test_errors_abs, label = 'Validation Error_abs')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Error')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'Train_curve{time.time():.4f}.png')
    plt.show()
    return None

def save_loss_comparisons(all_targets, all_outputs, all_inputs,
                         losses, errors, errors_abs,
                         save_dir="data",
                         img_size=(8, 4),
                         dpi=150,
                         batch_size=100):
    """
    按loss排序保存所有样本对比图及元数据（内存优化版）
    参数：
        all_targets: 真实值数组 [num_samples, ...]
        all_outputs: 预测值数组 [num_samples, ...]
        all_inputs: 输入参数数组 [num_samples]
        losses: 损失值数组 [num_samples]
        errors: 相对误差数组 [num_samples]
        errors_abs: 绝对误差数组 [num_samples]
        save_dir: 保存目录路径
        img_size: 单图尺寸 (宽, 高)
        dpi: 图片分辨率
        batch_size: 分批处理的大小，控制内存使用
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 按loss升序排序
    sorted_indices = np.argsort(losses)
    total_samples = len(sorted_indices)
    
    # 准备元数据CSV
    csv_path = os.path.join(save_dir, "metadata.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        csv_writer.writerow([
            'Global Rank', 'Sample Index', 'Loss Value',
            'Relative Error', 'Absolute Error', 'Input Parameters',
            'Image Filename'
        ])
        
        # 分批处理
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_indices = sorted_indices[batch_start:batch_end]
            wavelengths = np.linspace(1500, 1600, 301)
            # 处理当前批次
            for i, idx in enumerate(batch_indices, batch_start + 1):
                # 创建新figure
                fig = plt.figure(figsize=img_size)
                ax = fig.add_subplot(111)
                
                # 绘制对比曲线
                ax.plot(wavelengths, all_targets[idx], 
                        label='True', 
                        linewidth=2,
                        color='#1f77b4')
                ax.plot(wavelengths, all_outputs[idx],
                        label='Pred', 
                        linestyle='--',
                        linewidth=1.5,
                        color='#ff7f0e')
                
                # 添加标注信息
                ax.set_title(f"Loss Rank #{i}\n(params=[{all_inputs[idx][0] * 100:.0f}, {all_inputs[idx][1] * 100:.0f}]nm)", 
                            pad=12, 
                            fontsize=12)
                ax.set_xlabel('WaveLength(nm)', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper right', fontsize=9)
                
                # 生成文件名
                img_filename = f"loss_rank_{i}.png"
                save_path = os.path.join(save_dir, img_filename)
                
                # 保存图像并立即释放内存
                fig.tight_layout()
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)  # 显式关闭图形释放内存
                
                # 写入元数据
                csv_writer.writerow([
                    i,
                    idx,
                    f"{losses[idx]:.10f}",
                    f"{errors[idx]:.10f}",
                    f"{errors_abs[idx]:.10f}",
                    str(all_inputs[idx]),
                    img_filename
                ])
                
            # 可选：显示进度
            #print(f"Processed {batch_end}/{total_samples} samples...")
            
    print(f"Saved {total_samples} samples to: {os.path.abspath(save_dir)}")

def plot_sample():     
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
        #transforms.Normalize(())
        ])
    model_choice = int(input('Please select a model to train, 1 for PNNreal, 2 for PNNimag, 3 for FullModel: '))
    while(model_choice != 1 and model_choice != 2 and model_choice != 3):
        model_choice = int(input('Invalid input, please input 1, 2 or 3!'))
    if model_choice != 3:
        if model_choice == 1:
            _, _, test_loader, _, _, _, input_size, output_size, dataname = data_preprocess.prepare_data_PNN(batch_size=1, transform=transform, subscale=False)
            model = my_model.PNN(input_size=input_size, output_size=output_size, origin=False).to(device)
            prefix = f'PNNreal_{dataname}'
            model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
            if model_dict is not None:
                model.load_state_dict(model_dict)
        else:
            _, _, _, _, _, test_loader, input_size, output_size, dataname = data_preprocess.prepare_data_PNN(batch_size=1, transform=transform, subscale=False)
            model = my_model.PNN(input_size=input_size, output_size=output_size, origin=False).to(device)
            prefix = f'PNNimag_{dataname}'
            model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
            if model_dict is not None:
                model.load_state_dict(model_dict)
        print('Data Loaded!')
        print(model)
        losses = []
        errors = []
        errors_abs = []
        all_inputs = []
        all_targets = []
        all_outputs = []
        criterion = nn.MSELoss()
        mre = my_model.MRE()
        mare = my_model.MARE()
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                error = mre(outputs, targets)
                error_abs = mare(outputs, targets)
                losses.append(loss.item())
                errors.append(error.item())
                errors_abs.append(error_abs.item())
                all_inputs.append(inputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        losses = np.array(losses)
        errors = np.array(errors)
        errors_abs = np.array(errors_abs)
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        
        # 计算平均指标
        avg_loss = np.mean(losses)
        avg_error = np.mean(errors)
        avg_error_abs = np.mean(errors_abs)

        # 输出结果
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Error (MRE): {avg_error:.4f}")
        print(f"Average Absolute Error (MARE): {avg_error_abs:.4f}")
        
        '''plt.figure(figsize = (16, 5))
        plt.hist(losses, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Loss Values')
        plt.grid(True)
        plt.show()'''

        '''_, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))
        axs[0].hist(losses, bins = 50)
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        axs[0].grid(alpha = 0.6)
        mean_loss = np.mean(losses)
        axs[0].axvline(mean_loss)
        axs[0].set_title(f'MSE Distribution(mean = {mean_loss:.6f})')

        axs[1].hist(errors, bins = 50)
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        axs[1].grid(alpha = 0.6)
        p = 0.9
        q = np.quantile(errors, p)
        axs[1].axvline(q)
        axs[1].set_title(f'MRE Distribution(p{p} = {q:.6f})')

        axs[2].hist(errors_abs, bins = 50)
        axs[2].set_xlabel('Value')
        axs[2].set_ylabel('Frequency')
        axs[2].grid(alpha = 0.6)
        p = 0.9
        q = np.quantile(errors_abs, p)
        axs[2].axvline(q)
        axs[2].set_title(f'MARE Distribution(p{p} = {q:.6f})')

        plt.tight_layout()
        plt.show()
        
        def get_extreme_indices(data, num=8):
            sorted_indices = np.argsort(data)
            max_indices = sorted_indices[-num:][::-1]  # 从大到小
            min_indices = sorted_indices[:num]        # 从小到大
            return max_indices, min_indices
        loss_max_idx, loss_min_idx = get_extreme_indices(losses)
        error_max_idx, error_min_idx = get_extreme_indices(errors)
        error_abs_max_idx, error_abs_min_idx = get_extreme_indices(errors_abs)
        idx_list:list = [loss_max_idx, loss_min_idx, error_max_idx, error_min_idx, error_abs_max_idx, error_abs_min_idx]
        name_list:list = ['loss_max', 'loss_min', 'error_max', 'error_min', 'error_abs_max', 'error_abs_min']
        for j in range(len(idx_list)):
            _, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (15, 9))
            for i in range(8):
                i -= 4 # -4, -3, -2, -1, 0, 1, 2, 3
                axs[i // 4, i % 4].plot(all_targets[idx_list[j][i]], label = 'true')
                axs[i // 4, i % 4].plot(all_outputs[idx_list[j][i]], label = 'pred')
                axs[i // 4, i % 4].set_xlabel('Frequency')
                axs[i // 4, i % 4].set_ylabel('')
                axs[i // 4, i % 4].set_title(f'{name_list[j]}_{all_inputs[idx_list[j][i]]}')
                axs[i // 4, i % 4].legend()
            plt.tight_layout()
            plt.show()'''
        
        if model_choice == 1:
            save_loss_comparisons(all_targets, all_outputs, all_inputs,
                            losses, errors, errors_abs,
                            save_dir=os.path.join('data', f'{dataname}', 'PNN', 'real'),
                            img_size=(8, 4),
                            dpi=150)
        else:
            save_loss_comparisons(all_targets, all_outputs, all_inputs,
                            losses, errors, errors_abs,
                            save_dir=os.path.join('data', f'{dataname}', 'PNN', 'imag'),
                            img_size=(8, 4),
                            dpi=150) 
        
    else:
        train_loader, val_loader, test_loader, param_size, spectra_size, dataname = data_preprocess.prepare_data_Full(batch_size = 1, transform = transform, subscale = False)
        PNNreal = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNreal_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNreal_{dataname} ready for testing.')
        PNNreal.load_state_dict(model_dict)
        PNNimag = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNimag_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNimag_{dataname} ready for testing.')
        PNNimag.load_state_dict(model_dict)
        model = my_model.Fullmodel(input_size = spectra_size, output_size = param_size, PNNreal = PNNreal, PNNimag = PNNimag, origin = False).to(device)
        prefix = f'Fullmodel_{dataname}'
        model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
        if model_dict is not None:
            model.load_state_dict(model_dict)
        print('Data Loaded!')
        print(model)
        losses = []
        errors = []
        errors_abs = []
        all_params = []
        all_targets = []
        all_outputs = []
        criterion = my_model.custom_criterion(min_val = 0, max_val = 0, device = device, dataname = dataname, mode = 'mean')
        mre = my_model.MRE()
        mare = my_model.MARE()
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                params, outputs = model(inputs)
                loss = criterion((params, outputs), targets)
                error = mre(outputs, targets)
                error_abs = mare(outputs, targets)
                losses.append(loss.item())
                errors.append(error.item())
                errors_abs.append(error_abs.item())
                all_params.append(params.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        losses = np.array(losses)
        errors = np.array(errors)
        errors_abs = np.array(errors_abs)
        all_params = np.concatenate(all_params, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        # 计算平均指标
        avg_loss = np.mean(losses)
        avg_error = np.mean(errors)
        avg_error_abs = np.mean(errors_abs)

        # 输出结果
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Error (MRE): {avg_error:.4f}")
        print(f"Average Absolute Error (MARE): {avg_error_abs:.4f}")

        '''plt.figure(figsize = (16, 5))
        plt.hist(losses, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Loss Values')
        plt.grid(True)
        plt.show()'''
        
        '''_, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))
        axs[0].hist(losses, bins = 50)
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        axs[0].grid(alpha = 0.6)
        mean_loss = np.mean(losses)
        axs[0].axvline(mean_loss)
        axs[0].set_title(f'MSE Distribution(mean = {mean_loss:.6f})')

        axs[1].hist(errors, bins = 50)
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        axs[1].grid(alpha = 0.6)
        p = 0.9
        q = np.quantile(errors, p)
        axs[1].axvline(q)
        axs[1].set_title(f'MRE Distribution(p{p} = {q:.6f})')

        axs[2].hist(errors_abs, bins = 50)
        axs[2].set_xlabel('Value')
        axs[2].set_ylabel('Frequency')
        axs[2].grid(alpha = 0.6)
        p = 0.9
        q = np.quantile(errors_abs, p)
        axs[2].axvline(q)
        axs[2].set_title(f'MARE Distribution(p{p} = {q:.6f})')

        plt.tight_layout()
        plt.show()
        
        def get_extreme_indices(data, num=8):
            sorted_indices = np.argsort(data)
            max_indices = sorted_indices[-num:][::-1]  # 从大到小
            min_indices = sorted_indices[:num]        # 从小到大
            return max_indices, min_indices
        loss_max_idx, loss_min_idx = get_extreme_indices(losses)
        error_max_idx, error_min_idx = get_extreme_indices(errors)
        error_abs_max_idx, error_abs_min_idx = get_extreme_indices(errors_abs)
        idx_list:list = [loss_max_idx, loss_min_idx, error_max_idx, error_min_idx, error_abs_max_idx, error_abs_min_idx]
        name_list:list = ['loss_max', 'loss_min', 'error_max', 'error_min', 'error_abs_max', 'error_abs_min']
        for j in range(len(idx_list)):
            _, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (15, 9))
            for i in range(8):
                i -= 4 # -4, -3, -2, -1, 0, 1, 2, 3
                axs[i // 4, i % 4].plot(all_targets[idx_list[j][i]], label = 'true')
                axs[i // 4, i % 4].plot(all_outputs[idx_list[j][i]], label = 'pred')
                axs[i // 4, i % 4].set_xlabel('Frequency')
                axs[i // 4, i % 4].set_ylabel('')
                axs[i // 4, i % 4].set_title(f'{name_list[j]}_{all_params[idx_list[j][i]] * 100}')
                axs[i // 4, i % 4].legend()
            plt.tight_layout()
            plt.show()'''
        
        save_loss_comparisons(all_targets, all_outputs, all_params,
                         losses, errors, errors_abs,
                         save_dir=os.path.join('data', f'{dataname}', 'Full'),
                         img_size=(8, 4),
                         dpi=150)

def parse_log_file(file_path):
    epochs = []
    train_loss = []
    test_loss = []
    train_error = []
    test_error = []
    train_error_abs = []
    test_error_abs = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取所有数值
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(matches) >= 7:
                epochs.append(int(matches[0]))
                train_loss.append(float(matches[1]))
                test_loss.append(float(matches[2]))
                train_error.append(float(matches[3]))
                test_error.append(float(matches[4]))
                train_error_abs.append(float(matches[5]))
                test_error_abs.append(float(matches[6]))
    
    return {
        'epoch': epochs,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_error': train_error,
        'test_error': test_error,
        'train_error_abs': train_error_abs,
        'test_error_abs': test_error_abs
    }

def plot_metrics(data):
    epochs = data['epoch']
    
    # 创建2x3的子图布局
    plt.figure(figsize=(16, 5))
    
    # 绘制训练和测试损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data['train_loss'], linewidth = 2, label='Train Loss')
    plt.plot(epochs, data['test_loss'], linestyle = '--', linewidth = 1.5, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制最后的10%epochs的曲线
    index = int(0.9 * len(epochs))
    plt.subplot(1, 2, 2)
    plt.plot(epochs[index:], data['train_loss'][index:], linewidth = 1, label='Train Loss')
    plt.plot(epochs[index:], data['test_loss'][index:], linewidth = 1, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''# 绘制完整损失曲线
    axs[0, 0].plot(epochs, train_losses, label = 'Train Loss')
    axs[0, 0].plot(epochs, test_losses, label = 'Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    # 绘制最后的10%epochs的曲线
    index = int(0.9 * len(train_losses))
    axs[0, 1].plot(epochs[index:], train_losses[index:], label = 'Train Loss')
    axs[0, 1].plot(epochs[index:], test_losses[index:], label = 'Validation Loss')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()'''
    '''# 绘制训练和测试误差
    plt.subplot(2, 3, 2)
    plt.plot(epochs, data['train_error'], linewidth = 2, label='Train Error')
    plt.plot(epochs, data['test_error'], linestyle = '--', linewidth = 1.5, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Test Error')
    plt.legend()
    plt.grid(True)
    
    # 绘制训练和测试绝对误差
    plt.subplot(2, 3, 3)
    plt.plot(epochs, data['train_error_abs'], label='Train Error (Abs)')
    plt.plot(epochs, data['test_error_abs'], label='Validation Error (Abs)')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Error')
    plt.title('Training and Test Absolute Error')
    plt.legend()
    plt.grid(True)
    
    # 单独绘制训练损失
    plt.subplot(2, 3, 4)
    plt.plot(epochs, data['train_loss'], 'r-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # 单独绘制测试损失
    plt.subplot(2, 3, 5)
    plt.plot(epochs, data['test_loss'], 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.grid(True)
    
    # 单独绘制绝对误差
    plt.subplot(2, 3, 6)
    plt.plot(epochs, data['train_error_abs'], 'r-', label='Train Abs Error')
    plt.plot(epochs, data['test_error_abs'], 'b-', label='Vlidation Abs Error')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Errors')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()'''

def plot_metrics1(data):
    epochs = data['epoch']
    total_epochs = len(epochs)
    split_idx = int(0.2 * total_epochs)  # 前20%epoch分界点
    final_idx = int(0.9 * total_epochs)   # 最后10%起始点

    plt.figure(figsize=(16, 6))
    plt.suptitle('Learning Curve Analysis', fontsize=14, y=1.05)  # 总标题

    # --- 主图：完整训练曲线 ---
    ax1 = plt.subplot(1, 2, 1)
    # 用颜色区分不同阶段
    plt.plot(epochs[:split_idx], data['train_loss'][:split_idx], 
             'b-', linewidth=2, alpha=0.8, label='Train Loss (Fast Convergence)')
    plt.plot(epochs[split_idx:], data['train_loss'][split_idx:], 
             'b-', linewidth=1.5, alpha=0.5, label='Train Loss (Slow Decline)')
    plt.plot(epochs, data['test_loss'], 
             'r--', linewidth=1.5, label='Validation Loss')
    
    # 标记关键阶段
    plt.axvline(x=epochs[split_idx], color='gray', linestyle=':', alpha=0.5)
    plt.text(epochs[split_idx], max(data['train_loss'])*0.9, 
             'End of Fast Phase', rotation=90, ha='right', va='top', fontsize=10)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Full Training Process')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(data['train_loss']) * 1.1)  # 限制y轴范围

    # --- 子图：最后10%细节 ---
    ax2 = plt.subplot(1, 2, 2)
    plt.plot(epochs[final_idx:], data['train_loss'][final_idx:], 
             'b-', linewidth=1.5, label='Train Loss (Final 10%)')
    plt.plot(epochs[final_idx:], data['test_loss'][final_idx:], 
             'r--', linewidth=1.5, label='Validation Loss (Final 10%)')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Final Convergence Details')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

def visualize_mag_spec_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
        #transforms.Normalize(())
        ])
    # 读取原始数据
    parameter, magnitude, phase, param_size, spectra_size, dataname = data_preprocess.read_file_to_numpy(subscale=False, origin=True)
    # 获取数据加载器（train/val/test）
    phase_radians = np.radians(phase)
    real = magnitude * np.cos(phase_radians)
    imag = magnitude * np.sin(phase_radians)
    dataset = data_preprocess.Dataset_transformed_Full(real = real, imag = imag, transform = transform)

    # 检查是否有已保存的索引
    indices_path = os.path.join('data', f'{dataname}.npz')
    if os.path.exists(indices_path):
        print("Loading saved indices...")
        indices = np.load(indices_path)
        train_idx = indices["train_idx"]
        val_idx = indices["val_idx"]
        test_idx = indices["test_idx"]
    else:
        print('NO!')
        return
    print(f'Train length: {len(train_idx)} | Validation length: {len(val_idx)} | Test length: {len(test_idx)}')

    # 创建 Subset 数据集
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 定义保存目录
    save_dir = f'C:\\Programme\\projects\\linux\\metasurface\\{dataname}\\mag_spec_comparison'
    os.makedirs(save_dir, exist_ok=True)
    # 定义绘图参数
    img_size = (10, 5)
    dpi = 150
    wavelengths = np.linspace(1500, 1600, 301)  # 波长范围是1500-1600nm，共301个点
    # 准备CSV文件记录元数据
    csv_path = os.path.join(save_dir, "metadata.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Loss', 'Image Filename', 'Dataset Split'
        ])
        # 遍历测试集
        PNNreal = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNreal_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNreal_{dataname} ready for testing.')
        PNNreal.load_state_dict(model_dict)
        PNNimag = my_model.PNN(input_size = param_size, output_size = spectra_size, origin = False).to(device)
        model_dict, time_stamp, epoch = my_model.get_model_dict(f'PNNimag_{dataname}', device)
        if model_dict == None:
            exit(f'There is no PNNimag_{dataname} ready for testing.')
        PNNimag.load_state_dict(model_dict)
        model = my_model.Fullmodel(input_size = spectra_size, output_size = param_size, PNNreal = PNNreal, PNNimag = PNNimag, origin = False).to(device)
        prefix = f'Fullmodel_{dataname}'
        model_dict, time_stamp, epoch = my_model.get_model_dict(prefix, device)
        if model_dict is not None:
            model.load_state_dict(model_dict)
        print('Data Loaded!')
        print(model)
        i = 1
        j = 1
        for loader, split_name in [(test_loader, 'test'), ]:#[(train_loader, 'train'), (val_loader, 'val'), (test_loader, 'test')]:
            losses = []
            errors = []
            errors_abs = []
            criterion = my_model.custom_criterion(min_val = 1.5, max_val = 6.5, device = device, dataname = None, mode = 'mean')
            mre = my_model.MRE()
            mare = my_model.MARE()
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(loader):
                    # 获取当前样本的全局索引
                    global_idx = test_idx[batch_idx]
                    # 计算预测曲线和参数
                    inputs, targets = inputs.to(device), targets.to(device)
                    params, outputs = model(inputs)
                    loss = criterion((params, outputs), targets)
                    error = mre(outputs, targets)
                    error_abs = mare(outputs, targets)
                    losses.append(loss.item())
                    errors.append(error.item())
                    errors_abs.append(error_abs.item())
                    # 提取对应的magnitude和spectra
                    loss_np = loss.cpu().numpy
                    params_np = params.cpu().squeeze().numpy()  # 模型输出的参数 [param1, param2, ...]
                    target_np = targets.cpu().squeeze().numpy()  # 目标曲线 [301,]
                    # 获取目标相位（使用全局索引）
                    target_phase = phase[global_idx]
                    # 1. 搜索候选参数：在parameter中找到所有满足 |param_i - candidate_i| <= 2 的参数
                    tolerance = 0.02  # 允许的偏差范围
                    mask = np.all(np.abs(parameter - params_np) <= tolerance, axis=1)  # 所有维度同时满足条件
                    candidate_indices = np.where(mask)[0]  # 符合条件的候选索引
                    if len(candidate_indices) > 0:
                        # 2. 计算每个候选的仿真曲线与target的MSE Loss
                        losses = []
                        for idx in candidate_indices:
                            sim_curve = magnitude[idx]  # 候选仿真曲线
                            loss = np.mean((sim_curve - target_np) ** 2)  # MSE Loss
                            losses.append(loss)
                        
                        # 3. 选择Loss最小的候选
                        best_idx = candidate_indices[np.argmin(losses)]
                        matched_mag = magnitude[best_idx]
                        matched_phase = phase[matched_idx]
                        check = 'Y'
                    else:
                        # 4. 如果没有候选，回退到全局最近邻
                        distances = np.linalg.norm(parameter - params_np, axis=1)
                        matched_idx = np.argmin(distances)
                        matched_mag = magnitude[matched_idx]
                        matched_phase = phase[matched_idx]
                        check = 'N'
                    '''# 创建绘图
                    prefix = 'Mag_only'
                    fig = plt.figure(figsize=img_size)
                    ax = fig.add_subplot(111)
                    # 绘制magnitude和spectra
                    ax.plot(wavelengths, target_np, label='Target', linewidth=2.5, color='#1f77b4')
                    ax.plot(wavelengths, matched_mag, label='Simulation', linewidth=2, color='#ff7f0e', linestyle='--')
                    ax.plot(wavelengths, outputs.cpu().squeeze().numpy(), label='Prediction', linewidth=1, color='#2ca02c', linestyle='--')
                    # 添加标注
                    ax.set_title(
                        f"Params: [{params_np[0] * 100:.0f}, {params_np[1] * 100:.0f}]nm",
                        pad=12, fontsize=12
                    )
                    ax.set_xlabel('Wavelength (nm)', fontsize=10)
                    ax.set_ylabel('Value', fontsize=10)
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.legend(loc='upper right', fontsize=9)'''
                    # 创建绘图 包含相位
                    prefix = 'Full'
                    fig, ax1 = plt.subplots(figsize=img_size)
                    
                    # 绘制幅度曲线（左轴）
                    ax1.plot(wavelengths, target_np, label='Target Magnitude', linewidth=2.5, color='#1f77b4')
                    ax1.plot(wavelengths, matched_mag, label='Simulation Magnitude', linewidth=2, color='#ff7f0e', linestyle='--')
                    ax1.plot(wavelengths, outputs.cpu().squeeze().numpy(), 
                            label='Prediction Magnitude', linewidth=1, color='#2ca02c', linestyle='--')
                    
                    # 设置左轴属性
                    ax1.set_xlabel('Wavelength (nm)', fontsize=10)
                    ax1.set_ylabel('Magnitude', fontsize=10)
                    ax1.grid(True, linestyle=':', alpha=0.6)
                    
                    # 创建右轴用于相位
                    ax2 = ax1.twinx()
                    ax2.plot(wavelengths, matched_phase, label='Simulation Phase', linewidth=1, color='#9467bd', linestyle='-.')
                    ax2.plot(wavelengths, target_phase, label='Target Phase', linewidth=2, color='#8c564b', linestyle=':')
                    
                    # 设置右轴属性
                    ax2.set_ylabel('Phase (degrees)', fontsize=10)
                    ax2.set_ylim(-180, 180)  # 设置相位范围为[-180, 180]
                    ax2.grid(True, linestyle=':', alpha=0.6)
                    
                    # 合并图例
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
                    
                    # 设置标题
                    ax1.set_title(
                        f"Params: [{params_np[0]*100:.0f}, {params_np[1]*100:.0f}]nm",
                        pad=12, fontsize=12
                    )
                    # 保存图像
                    if check == 'Y':
                        img_filename = f"{prefix}_{check}_{i}.png"
                        i += 1
                    else:
                        img_filename = f"{prefix}_{check}_{j}.png"
                        j += 1
                    save_path = os.path.join(save_dir, img_filename)
                    fig.tight_layout()
                    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    # 记录元数据
                    csv_writer.writerow([
                        loss_np,
                        img_filename,
                        split_name
                    ])
            
    print(f"Visualization saved to: {os.path.abspath(save_dir)}")

if __name__ == '__main__':
    #visualize_mag_spec_comparison()
    #plot_parameter_distribution()
    plot_sample()
    #save_image()
    '''file_list = ['Fullmodel_11_timestamp1746500232.8717.txt',
                 'Fullmodel_22_timestamp1746533052.1150.txt',
                 'PNNreal_11_timestamp1745844902.6213.txt',
                 'PNNimag_11_timestamp1745844900.8816.txt',
                 'PNNreal_22_timestamp1745980575.6689.txt',
                 'PNNimag_22_timestamp1746054681.3098.txt']
    for file in file_list:
        file_path = f'C:\\Programme\\projects\\linux\\metasurface\\records\\{file}'
        data = parse_log_file(file_path)
        plot_metrics1(data)'''
    pass