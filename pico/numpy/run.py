#!/usr/bin/env python3

import os
import sys
import cam
import numpy as np
import matplotlib.pyplot as plt

#fname = '/royale/videos/royale_20180305_215512.rrf'
#fname = '/royale/videos/royale_20180308_224849.rrf'
#fname = '/royale/data/trial_2_action_5.rrf'

for trial in range(3):
    for action in range(10):
        in_path = os.path.join('/royale', 'data', 
            'trial_{}_action_{}.rrf'.format(trial, action))
        out_path = os.path.join('./data', 
            'trial_{}_action_{}.npy'.format(trial, action))
        c = cam.load(in_path)
        print('Extracting:', in_path)
        vid_list = []
        for modality in c:
            frame_list = []
            for frame in modality:
                frame = frame.T
                frame = np.flipud(frame)
                # If action = 0
                if action == 0:
                    frame = np.fliplr(frame)
                frame -= frame.min()
                frame /= frame.max()
                frame *= 255.0
                frame = frame.astype(np.uint8)
                frame = frame[:215, :]
                plt.imshow(frame)#, cmap='gray')
                plt.pause(0.01)
                plt.clf()
                frame_list.append(frame)
            plt.show()
            sys.exit()
            vid_list.append(frame_list)
        vid = np.array(vid_list)
        #with open(out_path, 'w') as fp:
        np.save(out_path, vid)
