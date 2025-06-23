import os
import sys
import logging
import random
import torch
import torch.nn as nn
import argparse
import numpy as np
import pickle
import time
import utils

from config_utils import load_config
from datasets import get_datasets, get_nas_search_loaders
from diffevo import DiffEvo
from nas_201_api import NASBench201API as API
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("NAS201")
parser.add_argument('--data', type=str, default='pkl', help='location of the data corpus')
parser.add_argument('--dir', type=str, default=None, help='location of trials')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=5, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--valid_batch_size', type=int, default=1024, help='validation batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--pop_size', type=int, default=50, help='population size')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=5, help='number of cells for NAS201 network')
parser.add_argument('--max_nodes', type=int, default=4, help='maximim nodes in the cell for NAS201 network')
parser.add_argument('--dataset', type=str, default='ImageNet16-120', help='["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--api_path', type=str, default='NAS-Bench-102-v1_0-e61699.pth', help='The path to the NAS-Bench-201 API file')
parser.add_argument('--config_path', type=str, default='./configs/CIFAR.config', help='The config path.')
args = parser.parse_args()
def get_arch_score(api, arch_index, dataset, hp, acc_type):
    info = api.query_by_index(arch_index, hp=str(hp))
    return info.get_metrics(dataset, acc_type)['accuracy']


def train(model, train_queue, criterion, optimizer, gen):
    model.train()
    pop = population.get_population()
    pop_size = len(pop)
    if not isinstance(pop, list) or not all(hasattr(x, 'arch_parameters') for x in pop):
        raise ValueError("population参数类型不符合预期，其元素应包含arch_parameters属性")
    for step, (inputs, targets) in enumerate(train_queue):
        current_index = step % args.pop_size
        if current_index >= pop_size:
            raise ValueError(f"Current index {current_index} exceeds population size {pop_size}")
        model.update_alphas(pop[current_index].arch_parameters[0])  # 假设这里是正确更新模型参数的方式，根据你的实际情况调整
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
        pop[current_index].objs.update(loss.data.cpu().item(), n)
        pop[current_index].top1.update(prec1.data.cpu().item(), n)
        pop[current_index].top5.update(prec5.data.cpu().item(), n)

        if (step + 1) % 100 == 0:
            logging.info("[{} Generation]".format(gen))
            logging.info(
                "Using Training batch #{} for {}/{} architecture with loss: {}, prec1: {}, prec5: {}".format(step,
                                                                                                             step % args.pop_size,
                                                                                                             len(pop),
                                                                                                             pop[
                                                                                                                 step % args.pop_size].objs.avg,
                                                                                                             pop[
                                                                                                                 step % args.pop_size].top1.avg,
                                                                                                             pop[
                                                                                                                 step % args.pop_size].top5.avg))


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

# Setup logging and directories
DIR = f"diffevo201_search100-{time.strftime('%Y%m%d-%H%M%S')}-{args.dataset}"
if args.dir:
    DIR = os.path.join(args.dir, DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(os.path.join(DIR, 'runs'))

# Set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(args.gpu)


# Load NAS-Bench-201 API
api = API(args.api_path)

# Dataset preparation
if args.dataset == 'cifar10':
    dataset_api = 'cifar10-valid'
    acc_type = 'x-valid'
else:
    dataset_api = args.dataset
    acc_type = 'x-test'

train_data, valid_data, xshape, num_classes = get_datasets(name=args.dataset, root=args.data, cutout=-1)
logging.info(f"train data len: {len(train_data)}, valid data len: {len(valid_data)}, xshape: {xshape}, #classes: {num_classes}")
config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
_, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, args.dataset, 'configs', (args.batch_size, args.valid_batch_size), args.workers)
logging.info(f'search_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

# Initialize model
model = TinyNetwork(C=args.init_channels, N=args.num_cells, max_nodes=args.max_nodes, num_classes=num_classes, search_space='NAS-Bench-201', affine=False, track_running_stats=True)
model = model.to(device)
optimizer, _, criterion = get_optim_scheduler(parameters=model.get_weights(), config=config)
criterion = criterion.cuda()
logging.info(f'optimizer: {optimizer}, Criterion: {criterion}')

# Initialize population
population = Population(pop_size=args.pop_size, num_edges=model.get_alphas()[0].shape[0], device=device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

# Instantiate DiffEvo
de_optimizer = DiffEvo(num_step=10, density='kde', noise=1.0)
start = time.time()

def fitness_function(individuals, valid_queue, criterion, gen):
    model.eval()
    fitness_values = []
    for individual in individuals:
        model.update_alphas(individual.arch_parameters[0])
        avg_loss, avg_accuracy1 = validation(model, valid_queue, criterion, gen)
        fitness_values.append(avg_accuracy1)
    return torch.tensor(fitness_values, device=device)

for epoch in range(args.epochs):
    # Training phase
    logging.info(f"[INFO] Generation {epoch + 1} training with learning rate {scheduler.get_lr()[0]}")
    train(model, train_loader, criterion, optimizer, epoch + 1)
    scheduler.step()

    # Validation phase
    logging.info(f"[INFO] Evaluating Generation {epoch + 1}")
    validation(model, valid_loader, criterion, epoch + 1)

    # Update population using DiffEvo
    population_tensor = torch.stack([ind.arch_parameters[0] for ind in population.get_population()])
    optimized_population = de_optimizer.optimize(lambda x: fitness_function(x, valid_queue, criterion, epoch + 1), population_tensor)

    # Convert optimized result back to Population object
    new_population = [Individual(arch_params=params) for params in optimized_population]
    population = Population(new_population)

    # Log metrics
    for i, p in enumerate(population.get_population()):
        writer.add_scalar(f"pop_top1_{i + 1}", p.get_fitness(), epoch + 1)
        writer.add_scalar(f"pop_top5_{i + 1}", p.top5.avg, epoch + 1)
        writer.add_scalar(f"pop_obj_valid_{i + 1}", p.objs.avg, epoch + 1)

    # Save results
    last = time.time() - start_time
    logging.info(f"[INFO] {epoch + 1}/{args.epochs} epoch finished in {last / 60} minutes")
    utils.save(model, os.path.join(DIR, "weights", "weights.pt"))

writer.close()
last = time.time() - start
logging.info("[INFO] Total search time: {} hours".format(last / 3600))
