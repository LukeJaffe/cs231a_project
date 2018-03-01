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
parser.add_argument('--mode', type=str, choices=['train', 'extract'], default='train')
parser.add_argument('--model_path', type=str, default='./checkpoint/ckpt.t7')
parser.add_argument('--feat_path', type=str, default='./features/ckpt.t7')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((args.resolution, args.resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.mode == 'train':
    DatasetType = ImageDataset
elif args.mode == 'extract':
    DatasetType = VideoDataset

trainset = DatasetType(args.partition_dir, 'train', transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

valset = DatasetType(args.partition_dir, 'val', transform_test)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=0)

class Pass(torch.nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

print('==> Building model..')
if args.mode == 'train':
    net = Autoencoder()
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

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, _) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f'
            % (train_loss/(batch_idx+1)))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = net(inputs)
        loss = criterion(outputs, inputs)

        test_loss += loss.data[0]

        progress_bar(batch_idx, len(valloader), 'Loss: %.3f'
            % (test_loss/(batch_idx+1)))

    # Save checkpoint.
    """
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/epoch{}.t7'.format(epoch))
        best_acc = acc
    """

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
    if not os.path.isdir('features'):
        os.mkdir('features')
    torch.save(emb_dict, './features/{}.t7'.format(mode))


if __name__=='__main__':
    if args.mode == 'train':
        for epoch in range(start_epoch, start_epoch+200):
            train(epoch)
            test(epoch)
    elif args.mode == 'extract':
        extract(trainloader, 'train')
        extract(valloader, 'val')
