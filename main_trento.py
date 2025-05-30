# -*- coding: utf-8 -*-
"""
@CreatedDate:   2020/4/27 12:08
@Author: Pangpd(https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: lyh
"""
import os
import sys
import time

from dataload_trento import train_loader, test_loader
from utils.auxiliary import save_acc_loss
from utils.auxiliary import get_logger
from utils.hyper_pytorch import *
from datetime import datetime

import torch
import torch.nn.parallel
import warnings
warnings.filterwarnings('ignore')
from utils.start import test, train, predict, output_metric
from models.complexNet_trento import ComplexNet as SNN
# from models.complexNet_Trento_CSMS import ComplexNet as SNN
# from models.complexNet_Trento_2ccl import ComplexNet as SNN
# from models.complexNet_crossattention import ComplexNet as SNN
np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------------定义超参数--------------------------
data_path = os.path.join(os.getcwd(), 'data')  # 数据集路径

dataset = 'PU'  # 数据集
seed = 1014
epochs = 100

learn_rate = 0.0085
# learn_rate = 0.00001
momentum = 0.9
weight_decay = 0.0001
# class_number = 22

iter = 1
def main():
    # ----------------------定义日志格式---------------------------
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "logs")  # logs目录
    log_dir = os.path.join(log_path, time_str)  # log组根目录

    oa_list = []
    aa_list = []
    kappa_list = []
    each_acc_list = []
    train_time_list = []
    test_time_list = []

    torch.cuda.empty_cache()
    group_log_dir = os.path.join(log_dir, "Experiment_")  # logs组目录
    if not os.path.exists(group_log_dir):
        os.makedirs(group_log_dir)
    group_logger = get_logger(str(iter + 1), group_log_dir)
    random_state = seed + iter
    print('-------------------------------------------Iter %s----------------------------------' % (iter + 1))
    start(group_log_dir, logger=group_logger)

def start(group_log_dir, logger):
    print('进入main.py 中的start方法！')
    use_cuda = True

    # model = SNN(10, leak_mem=0.7, img_size=spatial_size, num_cls=class_number, input_dim=components) #SA最好精度在40个步长
    model = SNN(20,input_dim=15) #SA最好精度在40个步长

    print(model)
    model =model.cuda()

    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(model.parameters(), learn_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    best_oa = -1
    best_aa = -1
    best_kappa = -1
    best_each_acc = -1
    best_acc = -1
    # 定义两个数组,记录训练损失和验证损失
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    train_start_time = time.time()  # 返回当前的时间戳
    for epoch in range(epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)

            data = np.transpose(data, (0, 2, 3, 1))
            hsi = data[..., 0:15]
            hsi = np.transpose(hsi, (0, 3, 1, 2))
            lidar = data[..., 15:]
            lidar = np.transpose(lidar, (0, 3, 1, 2))

            labels = labels.to(device)
            # Forward pass
            outputs = model(hsi,lidar)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss, valid_acc , test_acc1, test_obj, tar_v, pre_v= test(test_loader, model, criterion, epoch, use_cuda)

        logger.info('Epoch: %03d   Train Loss: %f Train Accuracy: %f   Valid Loss: %f Valid Accuracy: %f' % (
            epoch, train_loss, train_acc, valid_loss, valid_acc))

        OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
        # print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE * 100, AA_TE * 100, Kappa_TE))
        # print("CA: ", CA_TE * 100)
        # logger.info('AA: %f, OA: %f, kappa: %f\n ' % (AA_TE * 100, OA_TE * 100, Kappa_TE))
        # logger.info('CA: %s \n' , CA_TE * 100)
        scheduler.step(train_loss)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        # save model
        if valid_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, group_log_dir + "/best_model.pth_Trento.tar")
            best_acc = valid_acc
            best_oa = OA_TE * 100
            best_aa = AA_TE * 100
            best_kappa = Kappa_TE
            best_each_acc = CA_TE * 100

    logger.info('best_AA: %f, best_OA: %f, best_kappa: %f\n ' % (best_aa, best_oa, best_kappa))
    logger.info('best_CA: %s \n', best_each_acc)

    # train_end_time = time.time()
    # checkpoint = torch.load(group_log_dir + "/best_model.pth_Trento.tar")
    # best_acc = checkpoint['best_acc']
    # start_epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    #
    # # 测试
    # test_start_time = time.time()
    # test_loss, test_acc, test_acc1, test_obj, tar_v, pre_v = test(test_loader, model, criterion, epoch, use_cuda)
    # OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
    # print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(OA_TE * 100, AA_TE * 100, Kappa_TE))
    # logger.info('AA: %f, OA: %f, kappa: %f\n '% (OA_TE * 100, AA_TE * 100, Kappa_TE))
    # test_end_time = time.time()
    # logger.info("Final:   Loss: %s  Accuracy: %s", test_loss, test_acc)
    #
    # train_time = train_end_time - train_start_time
    # test_time = test_end_time - test_start_time
    # # logger.debug('classification:\n %s\n confusion:\n%s\n ' % (classification, confusion))
    # logger.info("Train time:%s , Test time:%s", train_time, test_time)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 225))  # 每隔25个epoch更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()