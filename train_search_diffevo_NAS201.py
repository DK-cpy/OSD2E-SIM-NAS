import os
import sys
import logging
import random
import torch.nn as nn
import argparse
import numpy as np
import pickle
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import utils
import pandas as pd

from cell_operationsNAS201 import NAS_BENCH_201
from config_utils import load_config
from datasets import get_datasets, get_nas_search_loaders
from diffevo import DiffEvo201  # 使用扩散进化算法
from nas_201_api import NASBench201API as API
from populationNAS201 import *
from optimizers import get_optim_scheduler
from search_model_NAS201 import TinyNetwork
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

parser = argparse.ArgumentParser("NAS201")
parser.add_argument('--data', type=str, default='pkl', help='location of the data corpus')
parser.add_argument('--dir', type=str, default=None, help='location of trials')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--valid_batch_size', type=int, default=1024, help='validation batch size')
parser.add_argument('--epochs', type=int, default=2, help='num of training epochs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=5, help='gpu device id')
parser.add_argument('--tsize', type=int, default=10, help='Tournament size')
parser.add_argument('--num_elites', type=int, default=1, help='Number of Elites')
parser.add_argument('--mutate_rate', type=float, default=0.1, help='mutation rate')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--pop_size', type=int, default=50, help='population size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')

parser.add_argument('--num_cells', type=int, default=5, help='number of cells for NAS201 network')
parser.add_argument('--max_nodes', type=int, default=4, help='maximim nodes in the cell for NAS201 network')
parser.add_argument('--track_running_stats', action='store_true', default=False,
                    help='use track_running_stats in BN layer')
parser.add_argument('--dataset', type=str, default='ImageNet16-120', help='["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--api_path', type=str, default='NAS-Bench-102-v1_0-e61699.pth',
                    help='["cifar10", "cifar10-valid", "cifar100", "imagenet16-120"]')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--config_path', type=str, default='./configs/CIFAR.config', help='The config path.')
args = parser.parse_args()


def get_arch_score(api, arch_index, dataset, hp, acc_type):
    info = api.query_by_index(arch_index, hp=str(hp))
    return info.get_metrics(dataset, acc_type)['accuracy']


def train(model, train_queue, criterion, optimizer, gen):
    model.train()
    pop = population.get_population()
    for step, (inputs, targets) in enumerate(train_queue):
        current_index = step % len(pop)
        model.update_alphas(pop[current_index].arch_parameters[0])  # 更新模型参数
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # 记录准确率和损失
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        pop[current_index].objs.update(loss.data.cpu().item(), inputs.size(0))
        pop[current_index].top1.update(prec1.data.cpu().item(), inputs.size(0))
        pop[current_index].top5.update(prec5.data.cpu().item(), inputs.size(0))

        if (step + 1) % 100 == 0:
            logging.info("[{} Generation] Training batch #{} loss: {}, prec1: {}, prec5: {}".format(
                gen, step, loss.item(), prec1.item(), prec5.item()))



'''
def train(model, train_queue, criterion, optimizer, gen):
    print(type(population))  # 打印传入参数的类型，方便调试
    if not hasattr(population, 'get_population'):
        raise ValueError("传入的population参数不是期望的类型，没有get_population方法")
    model.train()
    #chr_population = population.get_population()  # 获取当前种群
    for step, (inputs, targets) in enumerate(train_queue):
        chr_population = population.get_population()  # 获取当前种群
        model.update_alphas(chr_population[step % args.pop_size].arch_parameters[0])
        discrete_alphas = model.discretize()
        _, df_max, _ = model.show_alphas_dataframe()
        assert np.all(np.equal(df_max.to_numpy(), discrete_alphas.cpu().numpy()))
        assert model.check_alphas(discrete_alphas)

        n = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        _, logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        population.get_population()[step % args.pop_size].objs.update(loss.data.cpu().item(), n)
        population.get_population()[step % args.pop_size].top1.update(prec1.data.cpu().item(), n)
        population.get_population()[step % args.pop_size].top5.update(prec5.data.cpu().item(), n)

        if (step + 1) % 100 == 0:
            logging.info("[{} Generation]".format(gen))
            logging.info("Using Training batch #{} for {}/{} architecture with loss: {}, prec1: {}, prec5: {}".format(
                step, step % args.pop_size, len(population.get_population()),
                population.get_population()[step % args.pop_size].objs.avg,
                population.get_population()[step % args.pop_size].top1.avg,
                population.get_population()[step % args.pop_size].top5.avg))

'''


def validation(model, valid_queue, criterion, gen):
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    for i in range(len(population.get_population())):
        valid_start = time.time()
        # 更新模型的架构参数
        model.update_alphas(population.get_population()[i].arch_parameters[0])
        discrete_alphas = model.discretize()
        _, df_max, _ = model.show_alphas_dataframe()
        assert np.all(np.equal(df_max.to_numpy(), discrete_alphas.cpu().numpy()))
        assert model.check_alphas(discrete_alphas)

        # 重置当前个体的目标值
        population.get_population()[i].objs.reset()
        population.get_population()[i].top1.reset()
        population.get_population()[i].top5.reset()

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(valid_queue):
                n = inputs.size(0)
                inputs = inputs.to(device)
                targets = targets.to(device)
                _, logits = model(inputs)
                loss = criterion(logits, targets)

                # 计算损失和准确率
                total_loss += loss.item()
                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                population.get_population()[i].objs.update(loss.data.cpu().item(), n)
                population.get_population()[i].top1.update(prec1.data.cpu().item(), n)
                population.get_population()[i].top5.update(prec5.data.cpu().item(), n)

                correct1 += prec1.item()
                correct5 += prec5.item()
                total += n

    avg_loss = total_loss / len(valid_queue)
    avg_accuracy1 = correct1 / total
    avg_accuracy5 = correct5 / total

    # 日志记录
    for i in range(len(population.get_population())):
        logging.info("[{} Generation] {}/{} finished with validation loss: {}, prec1: {}, prec5: {}".format(
            gen, i + 1, len(population.get_population()),
            population.get_population()[i].objs.avg,
            population.get_population()[i].top1.avg,
            population.get_population()[i].top5.avg))

    # 返回平均损失和准确率
    return avg_loss, avg_accuracy1  # 返回平均损失和 top1 准确率


DIR = "2-demo-diffevo201_search10-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
if args.dir is not None:
    if not os.path.exists(args.dir):
        utils.create_exp_dir(args.dir)
    DIR = os.path.join(args.dir, DIR)
else:
    DIR = os.path.join(os.getcwd(), DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# 初始化 TensorBoard
writer = SummaryWriter(os.path.join(DIR, 'runs'))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:{}".format(args.gpu))
cpu_device = torch.device("cpu")

torch.cuda.set_device(args.gpu)
cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

assert args.api_path is not None, 'NAS201 data path has not been provided'
api = API(args.api_path, verbose=False)
logging.info(f'length of api: {len(api)}')

# 配置数据集和数据加载器
if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
assert args.dataset in datasets, 'Incorrect dataset'
if args.cutout:
    train_data, valid_data, xshape, num_classes = get_datasets(name=args.dataset, root=args.data, cutout=args.cutout)
else:
    train_data, valid_data, xshape, num_classes = get_datasets(name=args.dataset, root=args.data, cutout=-1)
logging.info(
    "train data len: {}, valid data len: {}, xshape: {}, #classes: {}".format(len(train_data), len(valid_data), xshape,
                                                                              num_classes))

config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
logging.info(f'config: {config}')
_, train_loader, valid_loader = get_nas_search_loaders(train_data=train_data, valid_data=valid_data,
                                                       dataset=args.dataset,
                                                       config_root='configs',
                                                       batch_size=(args.batch_size, args.valid_batch_size),
                                                       workers=args.workers)
train_queue, valid_queue = train_loader, valid_loader
logging.info('search_loader: {}, valid_loader: {}'.format(len(train_queue), len(valid_queue)))

# 模型初始化
model = TinyNetwork(C=args.init_channels, N=args.num_cells, max_nodes=args.max_nodes,
                    num_classes=num_classes, search_space=NAS_BENCH_201, affine=False,
                    track_running_stats=args.track_running_stats)
model = model.to(device)

optimizer, _, criterion = get_optim_scheduler(parameters=model.get_weights(), config=config)
criterion = criterion.cuda()
logging.info(f'optimizer: {optimizer}\nCriterion: {criterion}')

# 记录初始化的架构
best_arch_per_epoch = []

arch_str = model.genotype().tostr()
arch_index = api.query_index_by_arch(model.genotype())
if args.dataset == 'cifar10':
    test_acc = get_arch_score(api, arch_index, 'cifar10', 200, acc_type)
    valid_acc = get_arch_score(api, arch_index, 'cifar10-valid', 200, val_acc_type)
    writer.add_scalar("test_acc", test_acc, 0)
    writer.add_scalar("valid_acc", valid_acc, 0)
else:
    test_acc = get_arch_score(api, arch_index, args.dataset, 200, acc_type)
    valid_acc = get_arch_score(api, arch_index, args.dataset, 200, val_acc_type)
    writer.add_scalar("test_acc", test_acc, 0)
    writer.add_scalar("valid_acc", valid_acc, 0)
tmp = (arch_str, test_acc, valid_acc)
best_arch_per_epoch.append(tmp)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
logging.info(f'Scheduler: {scheduler}')

# 创建种群
population = Population(pop_size=args.pop_size, num_edges=model.get_alphas()[0].shape[0], device=device)
# 初始化种群张量（这个代码在创建 Population 实例之后立即执行）
initial_population_tensor = population.get_population_tensor()

logging.info(f'torch version: {torch.__version__}, torchvision version: {torch.__version__}')
logging.info("gpu device = {}".format(args.gpu))
logging.info("args =  %s", args)
logging.info("[INFO] Using diffusion evolution algorithm")
de = DiffEvo201(num_step=20, density='kde', noise=1.0)

start = time.time()


def fitness_function(individuals, valid_queue, criterion, gen, population_instance):
    model.eval()
    fitness_values = []
    with torch.no_grad():
        avg_loss, avg_accuracy = validation(model, valid_queue, criterion, gen)
        fitness_values = [avg_accuracy] * len(individuals)
    return torch.tensor(fitness_values)


start_time = time.time()
for epoch in range(args.epochs):
    # 训练整个种群
    logging.info("[INFO] Generation {} training with learning rate {}".format(epoch + 1, scheduler.get_lr()[0]))
    start_time = time.time()
    print("Before train function call, population type:", type(population))
    train(model, train_queue, criterion, optimizer, epoch + 1)
    print("After train function call, population type:", type(population))
    logging.info("[INFO] Training finished in {} minutes".format((time.time() - start_time) / 60))
    torch.save(model.state_dict(), "diffevo-50_nas201_model.pt")
    scheduler.step()

    logging.info("[INFO] Evaluating Generation {}".format(epoch + 1))
    validation(model, valid_queue, criterion, epoch + 1)

    # 根据适应度对种群进行排序
    population.pop_sort()

    for i, p in enumerate(population.get_population()):
        writer.add_scalar("pop_top1_{}".format(i + 1), p.get_fitness(), epoch + 1)
        writer.add_scalar("pop_top5_{}".format(i + 1), p.top5.avg, epoch + 1)
        writer.add_scalar("pop_obj_valid_{}".format(i + 1), p.objs.avg, epoch + 1)

    # 保存每一代的种群
    tmp = []
    for individual in population.get_population():
        tmp.append(tuple((individual.arch_parameters[0].cpu().numpy(), individual.get_fitness())))
    with open(os.path.join(DIR, "population_{}.pickle".format(epoch + 1)), 'wb') as f:
        pickle.dump(tmp, f)

    # 将最佳个体复制到模型中
    model.update_alphas(population.get_population()[0].arch_parameters[0])
    arch_str = model.genotype().tostr()
    arch_index = api.query_index_by_arch(model.genotype())
    if args.dataset == 'cifar10':
        test_acc = get_arch_score(api, arch_index, 'cifar10', 200, acc_type)
        valid_acc = get_arch_score(api, arch_index, 'cifar10-valid', 200, val_acc_type)
        writer.add_scalar("test_acc", test_acc, epoch + 1)
        writer.add_scalar("valid_acc", valid_acc, epoch + 1)
    else:
        test_acc = get_arch_score(api, arch_index, args.dataset, 200, acc_type)
        valid_acc = get_arch_score(api, arch_index, args.dataset, 200, val_acc_type)
        writer.add_scalar("test_acc", test_acc, epoch + 1)
        writer.add_scalar("valid_acc", valid_acc, epoch + 1)
    tmp = (arch_str, test_acc, valid_acc)
    best_arch_per_epoch.append(tmp)

    # 更新优化过程
    population_tensor = population.get_population_tensor()
    logging.info(f"Before optimization: {population_tensor.size()}")

    optimized_population_tensor = de.optimize201(
        lambda individuals: fitness_function(individuals, valid_queue, criterion, epoch + 1, population),
        population_tensor
    )
    logging.info(f"After optimization: {optimized_population_tensor.size()}")
    population.set_population_from_tensor(optimized_population_tensor)
    logging.info("[INFO] Optimization finished for generation {}".format(epoch + 1))
    '''
    # 记录并备份更新后的最佳架构
    alphas = population.get_population()[0].arch_parameters[0]
    expected_size = model.get_alphas()[0].size()
    print(f"Alphas size: {alphas.size()}, Expected size: {expected_size}")
    model.update_alphas(population.get_population()[0].arch_parameters[0])
    arch_str = model.genotype().tostr()
    arch_index = api.query_index_by_arch(model.genotype())
    test_acc = get_arch_score(api, arch_index, args.dataset, 200, acc_type)
    valid_acc = get_arch_score(api, arch_index, args.dataset, 200, val_acc_type)
    writer.add_scalar("test_acc", test_acc, epoch + 1)
    writer.add_scalar("valid_acc", valid_acc, epoch + 1)
    best_arch_per_epoch.append((arch_str, test_acc, valid_acc))
    '''
    torch.save(model.state_dict(), os.path.join(DIR, "weights", "model_{}.pt".format(epoch + 1)))
    logging.info("[INFO] Saved model for generation {}".format(epoch + 1))

writer.close()

last = time.time() - start
logging.info("[INFO] {} hours".format(last / 3600))

# for i in range(len(population.get_population())):
# logging.info(f'[INFO] {i}-th Best Architecture after the search: {best_arch_per_epoch[i]}')
# logging.info(f'length best_arch_per_epoch: {len(best_arch_per_epoch)}')
# with open(os.path.join(DIR, "best_architectures.pickle"), 'wb') as f:
#   pickle.dump(best_arch_per_epoch, f)

# 遍历并记录最佳架构
for i in range(len(best_arch_per_epoch)):
    logging.info(f'[INFO] {i}-th Best Architecture after the search: {best_arch_per_epoch[i]}')
