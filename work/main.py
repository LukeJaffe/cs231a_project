#!/usr/bin/env python3

import math

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
import collections

from torch.autograd import Variable

from utils import progress_bar
from dataset import ImageDataset, VideoDataset, PicoDataset

# Import models
import densenet
from models import FeatRNN, FullModel, Pass

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resolution', default=224, type=int, help='image resolution')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='num workers')
parser.add_argument('--num_frames', default=16, type=int, help='num frames in time slice')
parser.add_argument('--partition_dir', type=str, default='./partition')
parser.add_argument('--mode', type=str, choices=['train', 'test', 'extract', 'norm', 'lstm', 'full'], default='extract')
parser.add_argument('--model_dir', type=str, default='./trained')
parser.add_argument('--feature_dir', type=str, default='./features')
parser.add_argument('--lstm_path', type=str, default='./checkpoint/epoch190.t7')
parser.add_argument('--dep_metadata_path', type=str, default='./metadata/dep.t7')
parser.add_argument('--modality', type=str, choices=['rgb', 'dep', 'all', 'fuse', 'pico'], default='rgb')
parser.add_argument('--load_modality', type=str, choices=['rgb', 'dep', 'all', 'fuse', 'pico'], default='rgb')
parser.add_argument('--method', type=str, choices=['class', 'auto'], default='class')
parser.add_argument('--shuffle', default=1, type=int, help='shuffle')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_loss = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
if args.modality == 'fuse':
    rgb_train_transform = dep_train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((args.resolution, args.resolution)),
        #transforms.RandomCrop((args.resolution, args.resolution))
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914*0.2989+0.4822*0.5870+0.4465*0.1440,), 
            (0.2023*0.2989+0.1994*0.5870+0.2010*0.1440,)),
    ])

    rgb_test_transform = dep_test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((args.resolution, args.resolution)),
        #transforms.CenterCrop((args.resolution, args.resolution))
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914*0.2989+0.4822*0.5870+0.4465*0.1440,), 
            (0.2023*0.2989+0.1994*0.5870+0.2010*0.1440,)),
    ])
elif args.modality == 'pico':
    rgb_train_transform = rgb_test_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        #transforms.RandomCrop((args.resolution, args.resolution))
        transforms.ToTensor(),
        transforms.Normalize(
            (0.04275529876472071,), 
            (0.12779259683327668,)),
    ])

    dep_train_transform = dep_test_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        #transforms.CenterCrop((args.resolution, args.resolution))
        transforms.ToTensor(),
        transforms.Normalize(
            (0.15016975864264548,), 
            (0.15443158397547746,)),
    ])
else:
    rgb_train_transform = dep_train_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    rgb_test_transform = dep_test_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

if args.mode == 'lstm' or args.mode == 'full':
    lstm_flag = True
else:
    lstm_flag = False

if args.mode == 'train':
    eval_mode = False
else:
    eval_mode = True

if args.modality == 'pico':
    DatasetType = PicoDataset
else:
    DatasetType = VideoDataset

trainset = DatasetType(args.partition_dir, 'train', args.modality, 
        rgb_transform=rgb_train_transform, dep_transform=dep_train_transform, 
        num_frames=args.num_frames, eval_mode=eval_mode, lstm=lstm_flag)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

valset = DatasetType(args.partition_dir, 'val', args.modality, 
        rgb_transform=rgb_test_transform, dep_transform=dep_test_transform, 
        num_frames=args.num_frames, eval_mode=eval_mode, lstm=lstm_flag)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

testset = DatasetType(args.partition_dir, 'test', args.modality, 
        rgb_transform=rgb_test_transform, dep_transform=dep_test_transform, 
        num_frames=args.num_frames, eval_mode=eval_mode, lstm=lstm_flag)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

class FusedModel(torch.nn.Module):
    def __init__(self, rgb_model, dep_model, num_hidden=1000):
        super(FusedModel, self).__init__()
        self.rgb_model = rgb_model
        self.dep_model = dep_model
        self.fuser = torch.nn.Sequential(
            torch.nn.Linear(1024*2, num_hidden),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(num_hidden, 10)

    def forward(self, x):
        rgb_x = x[:, :, 0, :, :].unsqueeze(2)
        dep_x = x[:, :, 1, :, :].unsqueeze(2)
        rgb_out = self.rgb_model(rgb_x)
        dep_out = self.dep_model(dep_x)
        all_out = torch.cat([rgb_out, dep_out], dim=1)
        fuse_out = self.fuser(all_out)
        return self.classifier(fuse_out)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 2
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class RNN(torch.nn.Module):
    def __init__(self, num_lstm_feat=128, arch='dense'):
        super(RNN, self).__init__()

        if arch == 'dense':
            self.extracter = torchvision.models.densenet161(pretrained=False)
            self.extracter.features.conv0 = torch.nn.Conv2d(2, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.extracter.classifier = torch.nn.ReLU()
            self.lstm = torch.nn.LSTM(2208, num_lstm_feat, 1)
        elif arch == 'vgg11':
            self.extracter = torchvision.models.vgg11(pretrained=False)
            new = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            new_features = list(self.extracter.features)[1:]
            new_features.insert(0, new)
            self.extracter.features = torch.nn.Sequential(*new_features)
            self.extracter.classifier = torch.nn.ReLU()
            self.lstm = torch.nn.LSTM(25088, num_lstm_feat, 1)
        elif arch == 'squeeze':
            self.extracter = torchvision.models.squeezenet1_1(pretrained=False)
            new = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(2, 2))
            new_features = list(self.extracter.features)[1:]
            new_features.insert(0, new)
            self.extracter.features = torch.nn.Sequential(*new_features)
            self.extracter.classifier = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            self.extracter.num_classes = 13**2
            self.lstm = torch.nn.LSTM(13**2, num_lstm_feat, 1)
        elif arch == 'vgg':
            self.extracter = VGG('VGG11')
            self.lstm = torch.nn.LSTM(512, num_lstm_feat, 1)
        self.classifier = torch.nn.Sequential(
            #torch.nn.ReLU(),
            torch.nn.Linear(num_lstm_feat, 10)
        )

    def forward(self, x):
        #print(x.size())
        x = self.extracter(x)
        #print(x.size())
        x = x.unsqueeze(1)
        #print(x.size())
        x, _ = self.lstm(x)
        #print(x.size())
        x = x[:, -1, :]
        #print(x.size())
        x = self.classifier(x)
        #print(x.size())
        return x[-1, :].unsqueeze(0)

print('==> Building model..')
if args.mode == 'train':
    if args.modality == 'rgb':
        #net = torchvision.models.densenet161(pretrained=True)
        #net.classifier = torch.nn.Linear(2208, 10)
        net = densenet.densenet121(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=10)
    elif args.modality == 'dep':
        net = torchvision.models.densenet161(pretrained=True)
        net.classifier = torch.nn.Linear(2208, 10)
    elif args.modality == 'all':
        rgb_net = torchvision.models.densenet161(pretrained=True)
        dep_net = torchvision.models.densenet161(pretrained=True)
        rgb_net.classifier = Pass()
        dep_net.classifier = Pass()
        net = FusedModel(rgb_net, dep_net)
    elif args.modality == 'fuse':
        #net = RNN()
        #net = resnext.resnet152(sample_size=args.resolution, sample_duration=args.num_frames)
        #net = resnet.resnet18(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=10)
        net = densenet.densenet121(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=10)
    elif args.modality == 'pico':
        net = densenet.densenet121(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=10)
        #rgb_net = densenet.densenet121(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=10)
        #dep_net = densenet.densenet121(sample_size=args.resolution, sample_duration=args.num_frames, num_classes=10)
        #rgb_net.classifier = Pass()
        #dep_net.classifier = Pass()
        #net = FusedModel(rgb_net, dep_net)


elif args.mode == 'extract':
    model_path  = os.path.join(args.model_dir, '{}_{}.t7'.format(args.modality, args.method))
    checkpoint = torch.load(model_path)
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

elif args.mode == 'test' or args.mode == 'lstm':
    # Load spatial model
    model_path  = os.path.join(args.model_dir, '{}_{}.t7'.format(args.load_modality, args.method))
    checkpoint = torch.load(model_path)
    net = checkpoint['net']
    classifier = net.classifier
    net.classifier = Pass()
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    print('==> Loading CNN from epoch {}, acc={:.3f}'.format(epoch, acc))
    # Load temporal model
    checkpoint = torch.load(args.lstm_path)
    lstm = checkpoint['net']
    feat_mean = checkpoint['mean']
    feat_std = checkpoint['std']
    acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    print('==> Loading LSTM from epoch {}, acc={:2.3f}'.format(epoch, acc))
    if use_cuda:
        # Classifier
        classifier.cuda()
        classifier = torch.nn.DataParallel(classifier, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        # LSTM
        lstm.cuda()
        lstm = torch.nn.DataParallel(lstm, device_ids=range(torch.cuda.device_count()))
        lstm.benchmark = True

elif args.mode == 'full':
    cnn_path  = os.path.join(args.model_dir, '{}_{}.t7'.format(args.load_modality, args.method))
    full_model = FullModel(cnn_path, args.lstm_path)

if use_cuda and args.mode != 'full':
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if args.mode != 'full':
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\n==> Training (epoch {})'.format(epoch))
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
        #print(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        #print(targets)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #if batch_idx > 150:
        #    break

def test(epoch, dataloader):
    print('\n==> Testing (epoch {})'.format(epoch))
    global best_loss
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        targets = targets.squeeze()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if test_loss > best_loss:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        model_path  = os.path.join(args.model_dir, '{}_{}.t7'.format(args.modality, args.method))
        torch.save(state, model_path)
        best_loss = test_loss

def extract(dataloader, mode, max_chunk=100):
    net.eval()
    classifier.eval()
    correct = 0
    total = 0
    emb_dict = {}
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        if use_cuda:
            targets = targets.cuda()
        targets = Variable(targets, volatile=True)

        # Chunk the video slices for memory constraints
        #print('Input size:', inputs.size())
        chunk_emb, chunk_scores = [], []
        for chunk_idx in range(math.ceil(inputs.size(0)/max_chunk)):
            chunk = inputs[chunk_idx*max_chunk:chunk_idx*max_chunk+max_chunk].clone()
            #print('Chunk size:', chunk.size())
            if use_cuda:
                chunk = chunk.cuda()
            chunk = Variable(chunk, volatile=True)
            try:
                features = net(chunk)
            except RuntimeError:
                print('\nFailed batch, size:', inputs.size())
                continue
            scores = classifier(features)
            chunk_emb.append(features.cpu().data.clone())
            chunk_scores.append(scores.cpu().data.clone())
        emb_list.append(torch.cat(chunk_emb, dim=0))
        outputs = torch.cat(chunk_scores, dim=0)

        label_list.append(targets.cpu().data.clone())
        # Get argmax of score
        outputs = torch.sum(outputs, dim=0)

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data.cpu()).cpu().sum()

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('{} acc:'.format(mode), 100.*correct/total)
    emb_dict['features'] = emb_list
    emb_dict['labels'] = label_list
    emb_dict['acc'] = 100.*correct/total
    if not os.path.isdir(args.feature_dir):
        os.mkdir(args.feature_dir)
    torch.save(emb_dict, os.path.join(args.feature_dir, '{}_{}.t7'.format(args.modality, mode)))


def cnn_test(dataloader, max_chunk=100):
    net.eval()
    classifier.eval()
    correct = 0
    total = 0
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        if use_cuda:
            targets = targets.cuda()
        targets = Variable(targets, volatile=True)

        # Chunk the video slices for memory constraints
        #print('Input size:', inputs.size())
        chunk_emb, chunk_scores = [], []
        for chunk_idx in range(math.ceil(inputs.size(0)/max_chunk)):
            chunk = inputs[chunk_idx*max_chunk:chunk_idx*max_chunk+max_chunk].clone()
            #print('Chunk size:', chunk.size())
            if use_cuda:
                chunk = chunk.cuda()
            chunk = Variable(chunk, volatile=True)
            try:
                features = net(chunk)
            except RuntimeError:
                print('\nFailed batch, size:', inputs.size())
                continue
            scores = classifier(features)
            chunk_emb.append(features.cpu().data.clone())
            chunk_scores.append(scores.cpu().data.clone())
        emb_list.append(torch.cat(chunk_emb, dim=0))
        outputs = torch.cat(chunk_scores, dim=0)
        label_list.append(targets.cpu().data.clone())

        # Get argmax of score
        outputs = torch.sum(outputs, dim=0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data.cpu()).cpu().sum()

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('acc:', 100.*correct/total)

def lstm_test(dataloader, max_chunk=100):
    net.eval()
    classifier.eval()
    lstm.eval()
    correct = 0
    total = 0
    emb_list = []
    label_list = []
    conf_dict = collections.defaultdict(int)
    tot_dict = collections.defaultdict(int)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        if use_cuda:
            targets = targets.cuda()
        targets = Variable(targets, volatile=True)

        # Chunk the video slices for memory constraints
        #print('Input size:', inputs.size())
        chunk_emb, chunk_scores = [], []
        for chunk_idx in range(math.ceil(inputs.size(0)/max_chunk)):
            chunk = inputs[chunk_idx*max_chunk:chunk_idx*max_chunk+max_chunk].clone()
            #print('Chunk size:', chunk.size())
            if use_cuda:
                chunk = chunk.cuda()
            chunk = Variable(chunk, volatile=True)
            try:
                features = net(chunk)
            except RuntimeError:
                print('\nFailed batch, size:', inputs.size())
                continue
            scores = classifier(features)
            chunk_emb.append(features.cpu().data.clone())
            chunk_scores.append(scores.cpu().data.clone())
        emb = torch.cat(chunk_emb, dim=0)
        emb_list.append(emb)
        outputs = torch.cat(chunk_scores, dim=0)
        label_list.append(targets.cpu().data.clone())

        # Put output through LSTM
        inputs = emb
        inputs = (inputs - feat_mean) / feat_std
        if use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        inputs = inputs.unsqueeze(0)
        outputs = lstm(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        conf_dict[targets.cpu().data.numpy()[0]] += predicted.eq(targets.data).cpu().sum()
        tot_dict[targets.cpu().data.numpy()[0]] += targets.size(0)

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('acc:', 100.*correct/total)

    print('Class breakdown:')
    for c in tot_dict:
        print('{}: {:.2f}'.format(c, conf_dict[c]/tot_dict[c]))

def full_test(dataloader, max_chunk=100):
    full_model.eval()
    correct = 0
    total = 0
    emb_list = []
    label_list = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.squeeze()
        targets = targets.squeeze()
        for i, x in enumerate(inputs):
            print(x.size())
            sys.exit()
            pred = full_model(x)
            if pred == targets[i]:
                correct += 1
            total += 1

        progress_bar(batch_idx, len(dataloader), 'Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

    print('acc:', 100.*correct/total)

if __name__=='__main__':
    if args.mode == 'train':
        for epoch in range(start_epoch, start_epoch+1000):
            train(epoch)
            test(epoch, valloader)
    elif args.mode == 'test':
        cnn_test(valloader)
    elif args.mode == 'lstm':
        lstm_test(valloader)
    elif args.mode == 'full':
        full_test(valloader)
    elif args.mode == 'extract':
        extract(trainloader, 'train')
        extract(valloader, 'val')
        #extract(testloader, 'test')
