import os
import types
import json
import numpy as np
import torch
import torch.utils.data.dataset

from PIL import Image

# Take random time slice from video tensor
class RandomTimeSlice:
    def __init__(self, num_frames=16):
        self.num_frames = num_frames

    def __call__(self, vid_tsr):
        t0 = np.random.randint(0, vid_tsr.size(0)-self.num_frames)
        return vid_tsr[t0:t0+self.num_frames]

# Take random time slice from video tensor
class TimeSliceSet:
    def __init__(self, num_frames=16):
        self.num_frames = num_frames

    def __call__(self, vid_tsr):
        slice_list = []
        for t0 in range(vid_tsr.size(0)-self.num_frames):
            slice_list.append(vid_tsr[t0:t0+self.num_frames].unsqueeze(0))
        return torch.cat(slice_list, dim=0)


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

    return dep_tsr, label_tsr

def all_getitem(self, index):
    (rgb_mp4, rgb_path, dep_mp4, dep_path, person, background, illumination, pose, action) = self.data_list[index]

    rgb_img = Image.open(rgb_path)
    rgb_tsr = self.rgb_transform(rgb_img).unsqueeze(0)

    dep_img = Image.open(dep_path)
    dep_tsr = self.dep_transform(dep_img).unsqueeze(0)

    label_tsr = torch.LongTensor([int(action)-1])

    all_tsr = torch.cat([rgb_tsr, dep_tsr], dim=0)

    return all_tsr, label_tsr

def fuse_getitem(self, index):
    (rgb_mp4, rgb_path, dep_mp4, dep_path, person, background, illumination, pose, action) = self.data_list[index]

    rgb_img = Image.open(rgb_path)
    rgb_tsr = self.rgb_transform(rgb_img)

    dep_img = Image.open(dep_path)
    dep_tsr = self.dep_transform(dep_img)

    label_tsr = torch.LongTensor([int(action)-1])

    fuse_tsr = torch.cat([rgb_tsr, dep_tsr], dim=0)

    return fuse_tsr, label_tsr

class VideoDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, partition_dir, mode, modality, rgb_transform=None, dep_transform=None, num_frames=None, eval_mode=False, lstm=False):
        partition_file = '{}.json'.format(mode)
        partition_path = os.path.join(partition_dir, partition_file)
        with open(partition_path, 'r') as fp:
            self.data_list = json.load(fp)
        self.rgb_transform = rgb_transform
        self.dep_transform = dep_transform
        self.modality = modality
        if eval_mode:
            self.time_slice = TimeSliceSet(num_frames=num_frames)
        else:
            self.time_slice = RandomTimeSlice(num_frames=num_frames)

    def __getitem__(self, index):
        (rgb_mp4, rgb_dir, dep_mp4, dep_dir, person, background, illumination, pose, action) = self.data_list[index]

        if self.modality == 'rgb':
            path_list = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)]
            img_list = [Image.open(p) for p in path_list]
            img_tsr_list = [self.rgb_transform(img).unsqueeze(0) for img in img_list]
        elif self.modality == 'dep':
            path_list = [os.path.join(dep_dir, f) for f in os.listdir(dep_dir)]
            img_list = [Image.open(p) for p in path_list]
            img_tsr_list = [self.dep_transform(img).unsqueeze(0) for img in img_list]
        elif self.modality == 'fuse':
            path_list = [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)]
            rgb_img_list = [Image.open(p) for p in path_list]
            path_list = [os.path.join(dep_dir, f) for f in os.listdir(dep_dir)]
            dep_img_list = [Image.open(p) for p in path_list]
            rgb_tsr_list = [self.rgb_transform(rgb_img) for rgb_img in rgb_img_list]
            dep_tsr_list = [self.dep_transform(dep_img) for dep_img in dep_img_list]
            img_tsr_list = [torch.cat([rgb_tsr, dep_tsr], dim=0).unsqueeze(0) for rgb_tsr, dep_tsr in zip(rgb_tsr_list, dep_tsr_list)]

        vid_tsr = torch.cat(img_tsr_list)
        slice_tsr = self.time_slice(vid_tsr)

        label_tsr = torch.LongTensor([int(action)-1])

        if self.lstm:
            return slice_tsr, label_tsr.repeat(len(img_tsr))
        else:
            return slice_tsr, label_tsr

    def __len__(self):
        return len(self.data_list)

class ImageDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, partition_dir, mode, modality, rgb_transform=None, dep_transform=None, shuffle_videos=True):
        partition_file = '{}.json'.format(mode)
        partition_path = os.path.join(partition_dir, partition_file)
        with open(partition_path, 'r') as fp:
            data_list = json.load(fp)
        self.data_list = []
        if shuffle_videos:
            np.random.shuffle(data_list)
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
        elif modality == 'all':
            self._getitem_ = types.MethodType(all_getitem, self)
        elif modality == 'fuse':
            self._getitem_ = types.MethodType(fuse_getitem, self)

    def __getitem__(self, index):
        return self._getitem_(index)

    def __len__(self):
        return len(self.data_list)

class FeatureDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, feature_dir, modality, mode, feat_mean, feat_std):
        feature_file = '{}_{}.t7'.format(modality, mode)
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
        norm_tsr = (feat_tsr - self.feat_mean)/self.feat_std

        return norm_tsr, label_tsr.repeat(len(norm_tsr))

    def __len__(self):
        return len(self.feat_list)

class PicoDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, pico_dir, mode, modality, rgb_transform=None, dep_transform=None, num_frames=None, eval_mode=False, lstm=False):
        data_dir = os.path.join(pico_dir, mode)
        self.data_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.label_list = [os.path.splitext(f)[0].split('_')[3] for f in os.listdir(data_dir)]
        self.rgb_transform = rgb_transform
        self.dep_transform = dep_transform
        self.modality = modality
        if eval_mode:
            self.time_slice = TimeSliceSet(num_frames=num_frames)
        else:
            self.time_slice = RandomTimeSlice(num_frames=num_frames)
        self.lstm = lstm

    def __getitem__(self, index):
        action = self.label_list[index]
        vid_arr = np.load(self.data_list[index])
        vid_arr = vid_arr[:, :, :215, :]
        g_arr, d_arr = vid_arr

        rgb_img_list = [Image.fromarray(g) for g in g_arr]
        dep_img_list = [Image.fromarray(d) for d in d_arr]

        rgb_tsr_list = [self.rgb_transform(rgb_img) for rgb_img in rgb_img_list]
        dep_tsr_list = [self.dep_transform(dep_img) for dep_img in dep_img_list]
        img_tsr_list = [torch.cat([rgb_tsr, dep_tsr], dim=0).unsqueeze(0) for rgb_tsr, dep_tsr in zip(rgb_tsr_list, dep_tsr_list)]

        vid_tsr = torch.cat(img_tsr_list)
        slice_tsr = self.time_slice(vid_tsr)

        label_tsr = torch.LongTensor([int(action)])

        if self.lstm:
            return slice_tsr, label_tsr.repeat(len(slice_tsr))
        else:
            return slice_tsr, label_tsr

    def __len__(self):
        return len(self.data_list)
