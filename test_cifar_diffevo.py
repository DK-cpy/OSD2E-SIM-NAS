import os
import sys
import glob
import numpy as np
import torch
import utils  # 确保这个模块能在Notebook中找到
import logging
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network  # 确保这个模块能在Notebook中找到

import pickle

# Jupyter Notebook 中的参数设置
data = 'D:/DENAS-small/fasterdatasets'
model_path = 'C:/Users/19369/diffevo_model.pt'
batch_size = 96
report_freq = 50
gpu = 0
init_channels = 36
layers = 20
auxiliary = False
cutout = False
cutout_length = 16
drop_path_prob = 0.2
seed = 0
arch = 'DARTS'

# 日志设置
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        return  # 用 return 代替 sys.exit(1) 来避免整个 Notebook 中断

    np.random.seed(seed)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    logging.info('gpu device = %d' % gpu)

    genotype = eval("genotypes.%s" % arch)

    model = Network(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    utils.load(model, model_path, gpu)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_transforms_cifar10({'cutout': cutout, 'cutout_length': cutout_length})
    test_data = dset.CIFAR10(root=data, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = drop_path_prob
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)

def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

# 执行主函数
main()