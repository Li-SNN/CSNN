# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Katherine_Cao (https://github.com/Katherine-Cao/HSI_SNN)
"""

import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, accuracy_score

from utils import evaluate
import torch
import torch.nn.parallel

from utils.evaluate import AA_andEachClassAccuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (data,labels) in enumerate(trainloader):
        data = data.to(device)

        data = np.transpose(data, (0, 2, 3, 1))
        hsi = data[..., 0:64]
        hsi = np.transpose(hsi, (0, 3, 1, 2))
        lidar = data[..., 64:]
        lidar = np.transpose(lidar, (0, 3, 1, 2))

        labels = labels.to(device)
        outputs = model(hsi,lidar)
        loss = criterion(outputs, labels)  # CrossEntropyloss

        losses[batch_idx] = loss.item()
        accs[batch_idx] = evaluate.accuracy(outputs.data, labels.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.average(losses), np.average(accs)

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    print(matrix)
    # OA, AA_mean, Kappa, AA = cal_results(matrix[1:,1:])
    # print("haha")
    # OA, AA_mean, Kappa, AA = cal_results(matrix[:,1:12])
    OA, AA_mean, Kappa, AA = cal_results(matrix)

    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (data,targets) in enumerate(testloader):
            data = data.to(device)
            data = np.transpose(data, (0, 2, 3, 1))
            hsi = data[..., 0:20]
            hsi = np.transpose(hsi, (0, 3, 1, 2))
            lidar = data[..., 20:]
            lidar = np.transpose(lidar, (0, 3, 1, 2))

            targets = targets.to(device)

            outputs,spike_rates = model(hsi,lidar)
            for name, rate in spike_rates.items():
                print(f"{name} average firing rate: {rate:.4f}")

            losses[batch_idx] = criterion(outputs, targets).item()     # CrossEntropyLoss
            loss = criterion(outputs, targets)     # CrossEntropyLoss
            accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data, topk=(1,))[0].item()

            prec1, t, p = accuracy(outputs, targets, topk=(1,))
            n = hsi.shape[0]
            objs.update(loss.data, n)
            top1.update(prec1[0].data, n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())

    return np.average(losses), np.average(accs),top1.avg,objs.avg,tar,pre

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()

def predict(test_loader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda: inputs = inputs.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            hsi = inputs[..., 0:64]
            hsi = np.transpose(hsi, (0, 3, 1, 2))
            lidar = inputs[..., 64:]
            lidar = np.transpose(lidar, (0, 3, 1, 2))
            [predicted.append(a) for a in model(hsi,lidar).data.cpu().numpy()]
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))  # 1-149:0.1，150-200:0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
