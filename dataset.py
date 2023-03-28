import os
import time
import re
import shutil
import torch
import numpy as np
from torch.utils import data


class CuboidsDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, transform=None):
        """
        root_dir - directory with all the cuboids
        transform - transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.file_names = next(os.walk(root_dir))[2]
        if len(self.file_names) == 0:
            self.dirs = next(os.walk(root_dir))[1]
            for dir_ in self.dirs:
                files = os.listdir(os.path.join(root_dir, dir_))
                files = list(map(lambda x: os.path.join(root_dir, dir_, x), files))
                self.file_names.append(files)
            self.file_names = np.concatenate(self.file_names)
        self.transform = transform

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_path = os.path.join(self.root_dir, self.file_names[idx])
        sample = torch.from_numpy(np.load(file_path, mmap_mode='r'))
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def generator(loader):
    while True:
        for batch in loader:
            yield batch


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=True)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def data_transform(data, RVE=False):
    if RVE:
        train = []
        second_dimension = []
        data = data.copy()
        data = ''.join(data).split('},')
        for i in range(len(data)):
            if data[i][-2] != '}':
                elem_of_data = list(map(lambda x: 0 if '*' in x else int(float(x)), re.sub('[{}]', '', data[i]).split(',')))
                second_dimension.append(elem_of_data)
            else:
                elem_of_data = list(map(lambda x: 0 if '*' in x else int(float(x)), re.sub('[{}]', '', data[i]).split(',')))
                second_dimension.append(elem_of_data)
                train.append(second_dimension)
                second_dimension = []
        return np.array(train)
    else:
        train = []
        second_dimension = []
        data = data.copy()
        data = ''.join(data).split('},')
        for i in range(len(data)):
            if data[i][-2] != '}':
                elem_of_data = list(map(int, re.sub('[{}]', '', data[i]).split(',')))
                second_dimension.append(elem_of_data)
            else:
                elem_of_data = list(map(int, re.sub('[{}]', '', data[i]).split(',')))
                second_dimension.append(elem_of_data)
                train.append(second_dimension)
                second_dimension = []
        return np.array(train)
    

def prepare_dataset(path_train, path_val, shape, dtype='float32', RVE=False):
    """
    path_train - the path to the training files to be processed
    path_val - the path to the validation files to be processed
    shape - shape of data
    dtype - with what accuracy to keep
    """
    path_train = r'C:\Users\Evgeniy\Jupyter\Work\generated_cuboids\cuboids_64_spheresRVE\train'
    path_val = r'C:\Users\Evgeniy\Jupyter\Work\generated_cuboids\cuboids_64_spheresRVE\val'

    nx, ny, nz = shape
    X_train_filenames = []
    X_val_filenames = []

    for subdir, dirs, files in os.walk(path_train):
        for file in files:
            full_path = os.path.join(subdir, file)
            X_train_filenames.append(full_path)

    for subdir, dirs, files in os.walk(path_val):
        for file in files:
            full_path = os.path.join(subdir, file)
            X_val_filenames.append(full_path)

    X_train_filenames = np.array(X_train_filenames)
    X_val_filenames = np.array(X_val_filenames)


    try:
        os.mkdir(path_train + '_prepared')
    except FileExistsError:
        shutil.rmtree(path_train + '_prepared')
        os.mkdir(path_train + '_prepared')

    try:
        os.mkdir(path_val + '_prepared')
    except FileExistsError:
        shutil.rmtree(path_val + '_prepared')
        os.mkdir(path_val + '_prepared')

    scale = 0.5

    for i, file in enumerate(X_train_filenames):
        cuboid = []
        with open(file) as f:
            for line in f:
                cuboid.append(line[:-1])
        cuboid = data_transform(cuboid, RVE=RVE).reshape((nx, ny, nz))
        cuboid = np.expand_dims((np.array(cuboid) - scale) / scale, axis=0).astype(dtype)
        np.save(f'{path_train}_prepared/{i + 1}', cuboid)

    for i, file in enumerate(X_val_filenames):
        cuboid = []
        with open(file) as f:
            for line in f:
                cuboid.append(line[:-1])
        cuboid = data_transform(cuboid, RVE=RVE).reshape((nx, ny, nz))
        cuboid = np.expand_dims((np.array(cuboid) - scale) / scale, axis=0).astype(dtype)
        np.save(f'{path_val}_prepared/{i + 1}', cuboid)