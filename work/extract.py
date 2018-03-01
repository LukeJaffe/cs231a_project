#!/usr/bin/env python3

import os
import json
import subprocess
import collections

data_dir = '/home/username/data'
raw_dir = os.path.join(data_dir, 'raw')
proc_dir = os.path.join(data_dir, 'proc')
image_dir = os.path.join(proc_dir, 'images')
video_dir = os.path.join(proc_dir, 'videos')

### Format for data storage
# train is first 4 subjects, val is 1, test is 1
# Tuple:
# ( rgb_mp4, 
#   rgb_dir, 
#   dep_mp4, 
#   dep_dir, 
#   person, 
#   background, 
#   illumination, 
#   pose, 
#   action )
###

train_subjects = [
    'subject1',
    'subject2',
    'subject3',
    'subject4',
]

val_subjects = [
    'subject5',
]

test_subjects = [
    'subject6',
]

def prep(partition_dir, subject_list, mode):
    count = 0
    sep_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    for subject in subject_list:
        for modality in ['rgb', 'dep']:
            subject_modality = '{}_{}'.format(subject, 'rgb')
            subject_dir = os.path.join(raw_dir, subject_modality)
            for video_file in os.listdir(subject_dir):
                # Get name of video without file extension
                video_name = os.path.splitext(video_file)[0]
                # Extract metadata from file name
                (   _ , _, 
                    person, _, 
                    background, _, 
                    illumination, _, 
                    pose, _, 
                    action
                ) = video_name.split('_')
                # Build path to original video
                video_path = os.path.join(subject_dir, video_file)
                # Create new video
                new_video_file = '{}.mp4'.format(video_name)
                new_video_path = os.path.join(video_dir, new_video_file)
                subprocess.run(
                    ["ffmpeg", '-y', '-i',  video_path, new_video_path])
                # Extract frames to png files
                new_image_dir = os.path.join(image_dir, video_name)
                if not os.path.isdir(new_image_dir):
                    os.makedirs(new_image_dir)
                new_image_path = os.path.join(new_image_dir, 'frame%04d.png')
                subprocess.run(
                    ["ffmpeg", '-y', '-i',  video_path, new_image_path])
                # Create the data tuple
                key = (person, background, illumination, pose, action)
                sep_dict[key][modality] = [new_video_path, new_image_dir]
                # Increment count
                count += 1

    # Join RGB and depth data, create list of data elements
    data_list = []
    for key in sep_dict:
        (person, background, illumination, pose, action) = key
        rgb_mp4, rgb_dir = sep_dict[key]['rgb']
        dep_mp4, dep_dir = sep_dict[key]['dep']
        data = (rgb_mp4, rgb_dir, dep_mp4, dep_dir, person, background, illumination, pose, action)
        data_list.append(data)

    # Save off data list
    save_file = '{}.json'.format(mode)
    save_path = os.path.join(partition_dir, save_file)
    with open(save_path, 'w') as fp:
        json.dump(data_list, fp)

    print('Num videos:', count)
    return count

partition_dir = './partition'
if not os.path.isdir(partition_dir):
    os.makedirs(partition_dir)

train_count = prep(partition_dir, train_subjects, 'train')
val_count = prep(partition_dir, val_subjects, 'val')
test_count = prep(partition_dir, test_subjects, 'test')
print('\n\n')
print('train count:', train_count)
print('val count:', val_count)
print('test count:', test_count)
