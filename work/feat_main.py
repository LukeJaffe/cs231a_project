#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse

from torch.autograd import Variable

from utils import progress_bar
from dataset import FeatureDataset

import collections
from pprint import pprint

from models import FeatRNN


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--feature_dir', type=str, default='./features')
parser.add_argument('--modality', type=str, default='fuse')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
parser.add_argument('--model_path', type=str, default='./lstm/model.t7')
parser.add_argument('--model_dir', type=str, default='./lstm')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
normset = FeatureDataset(args.feature_dir, args.modality, 'train', None, None)
feat_mean, feat_std = normset.norm()

trainset = FeatureDataset(args.feature_dir, args.modality, 'train', feat_mean, feat_std)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)

valset = FeatureDataset(args.feature_dir, args.modality, 'val', feat_mean, feat_std)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=0)

#testset = FeatureDataset(args.feature_dir, args.modality, 'test', feat_mean, feat_std)
#testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)


if args.mode == 'train':
    print('==> Building model..')
    net = FeatRNN()
elif args.mode == 'test':
    checkpoint = torch.load(args.model_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    print('==> Loading model from epoch {}, acc={:.3f}'.format(epoch, acc))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.squeeze()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, dataloader, test=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    conf_dict = collections.defaultdict(int)
    tot_dict = collections.defaultdict(int)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        targets = targets.squeeze()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        #print(predicted.cpu().numpy()[0], targets.cpu().data.numpy()[0], '       ')
        correct += predicted.eq(targets.data).cpu().sum()
        conf_dict[targets.cpu().data.numpy()[0]] += predicted.eq(targets.data).cpu().sum()
        tot_dict[targets.cpu().data.numpy()[0]] += targets.size(0)

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    if not test:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.module if use_cuda else net,
                'mean': feat_mean,
                'std': feat_std,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(args.model_dir):
                os.mkdir(args.model_dir)
            torch.save(state, os.path.join(args.model_dir, 'model.t7'))
            best_acc = acc

    print('Class breakdown:')
    for c in tot_dict:
        print('{}: {:.2f}'.format(c, conf_dict[c]/tot_dict[c]))


if __name__=='__main__':
    if args.mode == 'train':
        for epoch in range(start_epoch, start_epoch+10000):
            train(epoch)
            test(epoch, valloader)
    elif args.mode == 'test':
        test(epoch, trainloader, test=True)
        test(epoch, valloader, test=True)
        #test(epoch, testloader)
