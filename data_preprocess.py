import scipy.io
from scipy import signal
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from typing import Tuple, List
def read_file_to_numpy(subscale:bool = True, origin: bool = False) -> Tuple[np.array, np.array, np.array, int, int, str]:
    """
    Get 'parameter', 'real' and 'imag' arrays from .mat file.
    
    :params:    [subscale] 论文中模型默认输出向量长度为31，设置为True时下采样数据集到31长度。
    :return:    返回np.array(结构参数)、np.array(实部)、np.array(虚部)、int(输入维度)、int(输出维度)、str(.mat文件名)。
    """
    print('Searching datafiles.')
    file_list : List = os.listdir(os.path.join(os.getcwd(), 'data'))
    datafile_list = list()
    for filename in file_list:
        if filename.endswith('.mat'):
            datafile_list.append(filename)
        elif filename.endswith('.csv') and filename.startswith('magnitude_data_'):
            datafile_list.append(filename)
    for i in range(len(datafile_list)):
        print(f'{i + 1}. {datafile_list[i]}')
    choice = int(input('Please choose a datafile to proceed: ')) - 1
    if datafile_list[choice].endswith('.mat'):
        dataname = datafile_list[choice][0:-4]
        mat_data = scipy.io.loadmat(os.path.join('data', datafile_list[choice]))
        parameter = mat_data['parameter'].astype(np.float32) # .mat默认为float64，torch默认接收float32
        real = mat_data['real'].astype(np.float32)
        imag = mat_data['imag'].astype(np.float32)
    elif datafile_list[choice].endswith('.csv'):
        dataname = datafile_list[choice][-6:-4]
        magnitude_data = np.genfromtxt(os.path.join('data', 'magnitude_data_' + dataname + '.csv'), delimiter=',', skip_header=1)
        phase_data = np.genfromtxt(os.path.join('data', 'phase_data_' + dataname + '.csv'), delimiter=',', skip_header=1)
        parameter = magnitude_data[:, :2].astype(np.float32)# .csv默认为float64，torch默认接收float32
        magnitude = magnitude_data[:, 2:].astype(np.float32)
        phase = phase_data[:, 2:].astype(np.float32)
        phase_radians = np.radians(phase) # 角度转换成弧度
        if origin:
            real = magnitude
            imag = phase

        else:
            real = magnitude * np.cos(phase_radians)
            imag = magnitude * np.sin(phase_radians)
    
            # 打印样本确认正确
            idx = 0
            print("Magnitude:", magnitude[idx, :5])
            print("Phase (rad):", np.radians(phase[idx, :5]))
            print("Real:", real[idx, :5])
            print("Imag:", imag[idx, :5])
            print("Recovered Mag:", np.sqrt(real[idx, :5] ** 2 + imag[idx, :5] ** 2))

    if subscale:
        def downsample(array:np.array, target_length:int = 31) -> Tuple[np.array, int]:
            """
            将数组[:, x]下采样至[:, 31]以满足论文模型需求

            :params:    [array]: 需要下采样的数组。
            :return:    返回下采样后的np.array(数组)、采样前长度int(origin_length)。
            """
            origin_length = array.shape[1]
            resampled = signal.resample(array, target_length, axis = 1)
            return resampled, origin_length
        real, origin_length = downsample(real)
        imag, _ = downsample(imag)
        print(f'实部虚部长度由{origin_length}下采样至{real.shape[1]}')
    input_size = parameter.shape[1]
    output_size = real.shape[1]
    print(f'Dataset length: {len(parameter)}, param_size: {input_size}, spectra_size: {output_size}')
    return parameter / 100, real, imag, input_size, output_size, dataname



def engineer_feature(parameter):
    pass

class Dataset_transformed_PNN(Dataset):
    def __init__(self, data:np.array, labels:np.array, transform:transforms.Compose = None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            label = self.transform(label)
        return sample, label
    
class Dataset_transformed_Full(Dataset):
    def __init__(self, real:np.array, imag:np.array, transform:transforms.Compose = None):
        self.real = real
        self.imag = imag
        self.transform = transform

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        spectra = np.sqrt(self.real[idx] ** 2 + self.imag[idx] ** 2) # 计算幅度
        if self.transform:
            spectra = self.transform(spectra)
        return spectra, spectra

def prepare_data_PNN(
    batch_size: int,
    transform: transforms.Compose = None,
    subscale: bool = True,
    indices_dir: str = "data",  # 索引保存路径
    ) -> tuple:
    # 读取数据
    parameter, real, imag, param_size, spectra_size, dataname = read_file_to_numpy(subscale=subscale)
    dataset_real = Dataset_transformed_PNN(parameter, real, transform)
    dataset_imag = Dataset_transformed_PNN(parameter, imag, transform)

    # 检查是否有已保存的索引
    indices_path = os.path.join(indices_dir, f'{dataname}.npz')
    if os.path.exists(indices_path):
        print("Loading saved indices...")
        indices = np.load(indices_path)
        train_idx = indices["train_idx"]
        val_idx = indices["val_idx"]
        test_idx = indices["test_idx"]
    else:
        # 随机划分索引（7:2:1）
        num_samples = len(dataset_real)
        indices = np.random.permutation(num_samples)
        train_idx = indices[:int(0.7 * num_samples)]
        val_idx = indices[int(0.7 * num_samples):int(0.9 * num_samples)]
        test_idx = indices[int(0.9 * num_samples):]

        # 保存索引
        np.savez(
            indices_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        print(f"Indices saved to {indices_path}")
    print(f'Train length: {len(train_idx)} | Validation length: {len(val_idx)} | Test length: {len(test_idx)}')
    # 创建 Subset 数据集
    train_dataset_real = Subset(dataset_real, train_idx)
    val_dataset_real = Subset(dataset_real, val_idx)
    test_dataset_real = Subset(dataset_real, test_idx)

    train_dataset_imag = Subset(dataset_imag, train_idx)
    val_dataset_imag = Subset(dataset_imag, val_idx)
    test_dataset_imag = Subset(dataset_imag, test_idx)

    # 创建 DataLoader
    train_loader_real = DataLoader(train_dataset_real, batch_size=batch_size, shuffle=True)
    val_loader_real = DataLoader(val_dataset_real, batch_size=batch_size, shuffle=False)
    test_loader_real = DataLoader(test_dataset_real, batch_size=batch_size, shuffle=False)

    train_loader_imag = DataLoader(train_dataset_imag, batch_size=batch_size, shuffle=True)
    val_loader_imag = DataLoader(val_dataset_imag, batch_size=batch_size, shuffle=False)
    test_loader_imag = DataLoader(test_dataset_imag, batch_size=batch_size, shuffle=False)

    return (
        train_loader_real,
        val_loader_real,
        test_loader_real,
        train_loader_imag,
        val_loader_imag,
        test_loader_imag,
        param_size,
        spectra_size,
        dataname,
    )


def prepare_data_Full(batch_size:int,
                      transform:transforms.Compose = None,
                      split_ratio:float = 0.8,
                      subscale = True,
                      indices_dir = 'data'
    ) -> tuple:
    # 读取数据
    _, real, imag, param_size, spectra_size, dataname = read_file_to_numpy(subscale = subscale)
    dataset = Dataset_transformed_Full(real = real, imag = imag, transform = transform)

    # 检查是否有已保存的索引
    indices_path = os.path.join(indices_dir, f'{dataname}.npz')
    if os.path.exists(indices_path):
        print("Loading saved indices...")
        indices = np.load(indices_path)
        train_idx = indices["train_idx"]
        val_idx = indices["val_idx"]
        test_idx = indices["test_idx"]
    else:
        # 随机划分索引（7:2:1）
        num_samples = len(dataset)
        indices = np.random.permutation(num_samples)
        train_idx = indices[:int(0.7 * num_samples)]
        val_idx = indices[int(0.7 * num_samples):int(0.9 * num_samples)]
        test_idx = indices[int(0.9 * num_samples):]

        # 保存索引
        np.savez(
            indices_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        print(f"Indices saved to {indices_path}")
    print(f'Train length: {len(train_idx)} | Validation length: {len(val_idx)} | Test length: {len(test_idx)}')

    # 创建 Subset 数据集
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        param_size,
        spectra_size,
        dataname,
    )

if __name__ == '__main__':
    while True:
        parameter, real, imag, _, _, _  = read_file_to_numpy(subscale = False)
        print(f'params: {parameter.shape}(max: {np.max(parameter)}, min: {np.min(parameter)}) real: {real.shape}[{np.max(real):.4f}, {np.min(real):.4f}], imag: {imag.shape}[{np.max(imag):.4f}, {np.min(imag):.4f}]')