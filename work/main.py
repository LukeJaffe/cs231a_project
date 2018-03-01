#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.autograd import Variable

from utils import progress_bar
from dataset import ImageDataset, VideoDataset

# Import models
from vgg import VGG
from densenet import DensenetHands

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resolution', default=224, type=int, help='image resolution')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--partition_dir', type=str, default='./partition')
parser.add_argument('--mode', type=str, choices=['train', 'extract', 'metadata'], default='extract')
parser.add_argument('--save_path', type=str, default='./trained/classifier.t7')
parser.add_argument('--model_path', type=str, default='./trained/classifier.t7')
parser.add_argument('--feature_dir', type=str, default='./features')
parser.add_argument('--dep_metadata_path', type=str, default='./metadata/dep.t7')
parser.add_argument('--modality', type=str, choices=['rgb', 'dep', 'all'], default='rgb')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
rgb_transform_train = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

rgb_transform_test = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dep_transform_train = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dep_transform_test = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.mode == 'train':
    DatasetType = ImageDataset
elif args.mode == 'extract':
    DatasetType = VideoDataset

trainset = DatasetType(args.partition_dir, 'train', args.modality, rgb_transform=rgb_transform_train, dep_transform=dep_transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

valset = DatasetType(args.partition_dir, 'val', args.modality, rgb_transform=rgb_transform_test, dep_transform=dep_transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)

class Pass(torch.nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x

class FusedModel(torch.nn.Module):
    def __init__(self, rgb_model, dep_model):
        super(FusedModel, self).__init__()
        self.rgb_model = rgb_model
        self.dep_model = dep_model

    def forward(self, x):
        return x

print('==> Building model..')
if args.mode == 'train':
    if args.modality == 'rgb':
        net = torchvision.models.densenet161(pretrained=True)
        net.classifier = torch.nn.Linear(2208, 10)
    elif args.modality == 'dep':
        net = torchvision.models.densenet161(pretrained=True)
        net.classifier = torch.nn.Linear(2208, 10)
elif args.mode == 'extract':
    checkpoint = torch.load(args.model_path)
    net = checkpoint['net']
    classifier = net.classifier
    net.classifier = Pass()
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    print('--> Loading from epoch {}, acc={:2.3f}'.format(epoch, acc))
    if use_cuda:
        classifier.cuda()
        classifier = torch.nn.DataParallel(classifier, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

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
        #inputs = inputs.squeeze()
        targets = targets.squeeze()
        #print(inputs.size(), targets.size())
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

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        #inputs = inputs.squeeze()
        targets = targets.squeeze()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('trained'):
            os.mkdir('trained')
        torch.save(state, args.save_path)
        best_acc = acc

def extract(dataloader, mode):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    emb_dict = {}
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        # Keep only 250 frames of video to fit on GPU
        if inputs.size(0) > 225:
            inputs = inputs[:225]
        targets = targets.squeeze()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        emb_list.append(outputs.cpu().data.clone())
        label_list.append(targets.cpu().data.clone())
        outputs = classifier(outputs)
        # Get argmax of score
        outputs = torch.sum(outputs, dim=0)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('{} acc:'.format(mode), 100.*correct/total)
    emb_dict['features'] = emb_list
    emb_dict['labels'] = label_list
    emb_dict['acc'] = 100.*correct/total
    if not os.path.isdir(args.feature_dir):
        os.mkdir(args.feature_dir)
    torch.save(emb_dict, os.path.join(args.feature_dir, '{}.t7'.format(mode)))


if __name__=='__main__':
    if args.mode == 'train':
        for epoch in range(start_epoch, start_epoch+200):
            train(epoch)
            test(epoch)
    elif args.mode == 'extract':
        extract(trainloader, 'train')
        extract(valloader, 'val')
