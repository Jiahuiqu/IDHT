import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader
import os
import random


class Dataset(nn.Module):
    def __init__(self, cemlabel, data, mode, channel=189, padding=2):
        super(Dataset_2, self).__init__()
        self.channel = channel
        self.mode = mode
        self.filename_data = data
        self.padding = padding
        self.cemlabel = cemlabel
        self.I = loadmat(os.path.join(self.filename_data))['data']
        self.h, self.w = self.I.shape[0], self.I.shape[1]
        random.seed(1)
        if self.mode == "train_1":
            self.whole_pointt = loadmat(os.path.join(self.cemlabel))['cem_1'].reshape(-1, 1)
            T_1 = list(np.where(self.whole_pointt == 1)[0])
            T_0 = list(np.where(self.whole_pointt == 0)[0])
            T1 = random.sample(T_1, np.ceil(len(T_1) * 0.5).astype('int32'))
            T0 = random.sample(T_0, np.ceil(len(T_0) * 0.07).astype('int32'))
            self.random_point = T1 + T0
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 1] = 1
        if self.mode == "train_2":
            self.whole_pointt = loadmat(os.path.join(self.cemlabel))['cem_21'].reshape(-1, 1)
            T_1 = list(np.where(self.whole_pointt == 1)[0])
            T_b = list(np.where(self.whole_pointt == 0)[0])
            T_2 = list(np.where(self.whole_pointt == 2)[0])
            T1 = random.sample(T_1, np.ceil(len(T_1) * 0.8).astype('int32'))
            T0 = random.sample(T_b, np.ceil(len(T_b) * 0.07).astype('int32'))
            T2 = random.sample(T_2, np.ceil(len(T_2) * 0.5).astype('int32'))
            self.random_point = T0 + T2 + T1
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 1] = 1
            self.label[self.whole_pointt == 2] = 2
        if self.mode == "train_3":
            self.whole_pointt = loadmat(os.path.join(self.cemlabel))['cem_321'].reshape(-1, 1)
            T_1 = list(np.where(self.whole_pointt == 1)[0])
            T_b = list(np.where(self.whole_pointt == 0)[0])
            T_2 = list(np.where(self.whole_pointt == 2)[0])
            T_3 = list(np.where(self.whole_pointt == 3)[0])
            T1 = random.sample(T_1, np.ceil(len(T_1) * 0.8).astype('int32'))
            T0 = random.sample(T_b, np.ceil(len(T_b) * 0.07).astype('int32'))
            T2 = random.sample(T_2, np.ceil(len(T_2) * 0.8).astype('int32'))
            T3 = random.sample(T_3, np.ceil(len(T_3) * 0.5).astype('int32'))
            self.random_point = T0 + T2 + T1 + T3
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 1] = 1
            self.label[self.whole_pointt == 2] = 2
            self.label[self.whole_pointt == 3] = 3
        if self.mode == "train_4":
            self.whole_pointt = loadmat(os.path.join(self.cemlabel))['cem_4321'].reshape(-1, 1)
            T_1 = list(np.where(self.whole_pointt == 1)[0])
            T_b = list(np.where(self.whole_pointt == 0)[0])
            T_2 = list(np.where(self.whole_pointt == 2)[0])
            T_3 = list(np.where(self.whole_pointt == 3)[0])
            T_4 = list(np.where(self.whole_pointt == 4)[0])
            T1 = random.sample(T_1, np.ceil(len(T_1) * 0.8).astype('int32'))
            T0 = random.sample(T_b, np.ceil(len(T_b) * 0.07).astype('int32'))
            T2 = random.sample(T_2, np.ceil(len(T_2) * 0.8).astype('int32'))
            T3 = random.sample(T_3, np.ceil(len(T_3) * 0.8).astype('int32'))
            T4 = random.sample(T_4, np.ceil(len(T_4) * 0.5).astype('int32'))
            self.random_point = T0 + T2 + T1 + T3 + T4
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 1] = 1
            self.label[self.whole_pointt == 2] = 2
            self.label[self.whole_pointt == 3] = 3
            self.label[self.whole_pointt == 4] = 4
        if self.mode == "test_1":
            self.whole_pointt = loadmat(os.path.join(self.filename_data))['label'].reshape(-1, 1)
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 1] = 1
            self.random_point = list(range(self.h*self.w))
        if self.mode == "test_2":
            self.whole_pointt = loadmat(os.path.join(self.filename_data))['label'].reshape(-1, 1)
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 2] = 2
            self.random_point = list(range(self.h*self.w))
        if self.mode == "test_3":
            self.whole_pointt = loadmat(os.path.join(self.filename_data))['label'].reshape(-1, 1)
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 3] = 3
            self.random_point = list(range(self.h*self.w))
        if self.mode == "test_4":
            self.whole_pointt = loadmat(os.path.join(self.filename_data))['label'].reshape(-1, 1)
            self.whole_pointt = self.whole_pointt.reshape(self.h, self.w)
            self.label = np.zeros_like(self.whole_pointt)
            self.label[self.whole_pointt == 4] = 4
            self.random_point = list(range(self.h*self.w))

    def __len__(self):
        return len(self.random_point)

    def __getitem__(self, index):

        original_i = int((self.random_point[index] / self.w))
        original_j = (self.random_point[index] - original_i * self.w)
        new_i = original_i + self.padding
        new_j = original_j + self.padding
        label = self.label[original_i, original_j]
        imgr = self.I[new_i, new_j, :].astype(np.float32)
        data = torch.from_numpy(imgr.reshape(imgr.shape[0], 1, 1))
        return data, label


