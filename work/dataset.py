import os
import types
import json
import torch
import torch.utils.data.dataset

from PIL import Image

class VideoDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, partition_dir, mode, modality, rgb_transform=None, dep_transform=None):
        partition_file = '{}.json'.format(mode)
        partition_path = os.path.join(partition_dir, partition_file)
        with open(partition_path, 'r') as fp:
            self.data_list = json.load(fp)
        self.transform = transform

    def __getitem__(self, index):
        (rgb_mp4, rgb_dir, dep_mp4, dep_dir, person, background, illumination, pose, action) = self.data_list[index]

        path_list = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)]
        img_list = [Image.open(p) for p in path_list]
        img_tsr_list = [self.transform(img).unsqueeze(0) for img in img_list]

        img_tsr = torch.cat(img_tsr_list)

        label_tsr = torch.LongTensor([int(action)-1])

        return img_tsr, label_tsr#.repeat(len(img_tsr))

    def __len__(self):
        return len(self.data_list)

def rgb_getitem(self, index):
    (rgb_mp4, rgb_path, dep_mp4, dep_path, person, background, illumination, pose, action) = self.data_list[index]

    rgb_img = Image.open(rgb_path)
    rgb_tsr = self.rgb_transform(rgb_img)
    label_tsr = torch.LongTensor([int(action)-1])

    return rgb_tsr, label_tsr

def dep_getitem(self, index):
    (rgb_mp4, rgb_path, dep_mp4, dep_path, person, background, illumination, pose, action) = self.data_list[index]

    dep_img = Image.open(dep_path)
    dep_tsr = self.dep_transform(dep_img)
    label_tsr = torch.LongTensor([int(action)-1])

    #print(dep_tsr.size())
    #print(dep_tsr.mean(dim=1).mean(dim=2))

    return dep_tsr, label_tsr

class ImageDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, partition_dir, mode, modality, rgb_transform=None, dep_transform=None):
        partition_file = '{}.json'.format(mode)
        partition_path = os.path.join(partition_dir, partition_file)
        with open(partition_path, 'r') as fp:
            data_list = json.load(fp)
        self.data_list = []
        for (rgb_mp4, rgb_dir, dep_mp4, dep_dir, person, background, illumination, pose, action) in data_list:
            for rgb_file, dep_file in zip(sorted(os.listdir(rgb_dir)), sorted(os.listdir(dep_dir))):
                rgb_path = os.path.join(rgb_dir, rgb_file)
                dep_path = os.path.join(dep_dir, dep_file)
                data = rgb_mp4, rgb_path, dep_mp4, dep_path, person, background, illumination, pose, action
                self.data_list.append(data)
                 
        self.rgb_transform = rgb_transform
        self.dep_transform = dep_transform

        if modality == 'rgb':
            self._getitem_ = types.MethodType(rgb_getitem, self)
        elif modality == 'dep':
            self._getitem_ = types.MethodType(dep_getitem, self)

    def __getitem__(self, index):
        return self._getitem_(index)

    def __len__(self):
        return len(self.data_list)

class FeatureDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, feature_dir, mode, feat_mean, feat_std):
        feature_file = '{}.t7'.format(mode)
        feature_path = os.path.join(feature_dir, feature_file)
        data_dict = torch.load(feature_path)
        self.feat_list = data_dict['features']
        self.label_list = data_dict['labels']
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def norm(self):
        feat_tsr = torch.cat(self.feat_list)
        feat_mean = torch.mean(feat_tsr, dim=0)
        feat_std = torch.std(feat_tsr, dim=0)
        feat_std[feat_std==0] = 1
        return feat_mean, feat_std

    def __getitem__(self, index):
        feat_tsr = self.feat_list[index]
        label_tsr = self.label_list[index]
        #norm_tsr = (feat_tsr - self.feat_mean)/self.feat_std

        return feat_tsr, label_tsr#.repeat(len(norm_tsr))

    def __len__(self):
        return len(self.feat_list)
