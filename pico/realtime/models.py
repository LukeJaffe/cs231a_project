import torch
import os
import sys
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

class Pass(torch.nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x

class FeatRNN(torch.nn.Module):
    def __init__(self, num_hidden=128):
        super(FeatRNN, self).__init__()

        self.lstm = torch.nn.LSTM(1024, num_hidden, 2, dropout=0.9)
        #self.lstm = torch.nn.LSTM(1024, num_hidden, 1)
        self.classifier = torch.nn.Linear(num_hidden, 10)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        #x = x.squeeze()
        #x = x[:, -1, :]
        x = self.act(x)
        #x = x.squeeze()
        #print(x.size())
        x = self.classifier(x)
        #print(x.size())
        return x.squeeze()


class FullModel(torch.nn.Module):
    def __init__(self, cnn_path, lstm_path, use_cuda=True, return_label=False):
        super(FullModel, self).__init__()
        self.use_cuda = use_cuda
        self.return_label = return_label
        self.label_list = [
            'Circle', 'Triangle', 'Up-down', 'Right-left', 'Wave', 'Z',
            'Cross', 'Comehere', 'Turn around', 'Pat'
        ]

        # Load spatial model
        checkpoint = torch.load(cnn_path)
        cnn = checkpoint['net']
        #classifier = net.classifier
        cnn.classifier = Pass()
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print('==> Loading CNN from epoch {}, acc={:.3f}'.format(epoch, acc))
        # Load temporal model
        checkpoint = torch.load(lstm_path)
        lstm = checkpoint['net']
        self.feat_mean = checkpoint['mean']
        self.feat_std = checkpoint['std']
        acc = checkpoint['acc']
        epoch = checkpoint['epoch']
        print('==> Loading LSTM from epoch {}, acc={:2.3f}'.format(epoch, acc))
        if use_cuda:
            # Classifier
            cnn.cuda()
            cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
            # LSTM
            lstm.cuda()
            lstm = torch.nn.DataParallel(lstm, device_ids=range(torch.cuda.device_count()))
            # Convert mean, std
            self.feat_mean = self.feat_mean.cuda()
            self.feat_std = self.feat_std.cuda()
            # CUDNN
            cudnn.benchmark = True

        self.feat_mean = Variable(self.feat_mean, volatile=True)
        self.feat_std = Variable(self.feat_std, volatile=True)

        self.cnn = cnn
        self.lstm = lstm

    def forward(self, x):
        x = x.unsqueeze(0)
        if self.use_cuda:
            x = x.cuda()
        x = Variable(x, volatile=True)
        f = self.cnn(x)
        f = (f - self.feat_mean) / self.feat_std
        f = f.unsqueeze(0)
        s = self.lstm(f)
        s = s.unsqueeze(0)
        _, p = torch.max(s.cpu().data, 1)
        i = p[0]
        if self.return_label:
            return self.label_list[i]
        else:
            return i
