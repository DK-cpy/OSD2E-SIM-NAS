import os
import sys
import logging
import random
import torch.nn as nn
import genotypes
import utils
import torch.utils
import numpy as np
from torch.autograd import Variable
from model_search import Network
import argparse
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from population import Population
from functools import partial
import pickle
from diffevo import DiffEvo  # 假设 DiffEvo 类位于 diffusion_evo.py 文件中

# 新增全局变量
best_accuracy = 0.0
best_model_path = 'best_teacher.pth'

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/fasterdatasets/cifar-100', help='location of the data corpus')
parser.add_argument('--dir', type=str, default=None, help='location of trials')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--tsize', type=int, default=10, help='Tournament size')
parser.add_argument('--num_elites', type=int, default=50, help='Number of Elites')
parser.add_argument('--mutate_rate', type=float, default=0.2, help='mutation rate')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--pop_size', type=int, default=50, help='population size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--warm_up', type=int, default=0, help='warm up epochs')
args = parser.parse_args()

def warm_train(model, train_queue, criterion, optimizer):
    model.train()
    for step, (inputs, targets) in enumerate(train_queue):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

def train(model, train_queue, criterion, optimizer, gen):
    model.train()
    for step, (inputs, targets) in enumerate(train_queue):
        model.copy_arch_parameters(population.get_population()[step % args.pop_size].get_arch_parameters())
        n = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        population.get_population()[step % args.pop_size].objs.update(loss.data, n)
        population.get_population()[step % args.pop_size].top1.update(prec1.data, n)
        population.get_population()[step % args.pop_size].top5.update(prec5.data, n)

        if (step + 1) % 100 == 0:
            logging.info("[{} Generation]".format(gen))
            logging.info(
                "Using Training batch #{} for {}/{} architecture with loss: {}, prec1: {}, prec5: {}".format(step,
                                                                                                             step % args.pop_size,
                                                                                                             len(population.get_population()),
                                                                                                             population.get_population()[
                                                                                                                 step % args.pop_size].objs.avg,
                                                                                                             population.get_population()[
                                                                                                                 step % args.pop_size].top1.avg,
                                                                                                             population.get_population()[
                                                                                                                 step % args.pop_size].top5.avg))

def validation(model, valid_queue, criterion, gen, population):
    model.eval()
    for i in range(len(population.get_population())):
        individual = population.get_population()[i]
        individual.objs.reset()
        individual.top1.reset()
        individual.top5.reset()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            discrete_alphas = utils.discretize(
                population.get_population()[step % population.get_population_size()].arch_parameters, device)
            model.copy_arch_parameters(discrete_alphas)
            n = inputs.size(0)
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            population.get_population()[step % population.get_population_size()].objs.update(loss.data, n)
            population.get_population()[step % population.get_population_size()].top1.update(prec1.data, n)
            population.get_population()[step % population.get_population_size()].top5.update(prec5.data, n)

    population.pop_sort()
    for i in range(args.pop_size):
        discrete_alphas = utils.discretize(population.get_population()[i].arch_parameters, device)
        model.copy_arch_parameters(discrete_alphas)
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        logging.info(
            "[{} Generation] {}/{} finished with validation loss: {}, prec1: {}, prec5: {}, scaling factor: {}, encoding: {}".format(
                gen, i + 1, len(population.get_population()),
                population.get_population()[i].objs.avg,
                population.get_population()[i].top1.avg,
                population.get_population()[i].top5.avg,
                population.get_population()[i].get_mutate_factor(),
                discrete_alphas))


import torch

def fitness_function(x_population, model, valid_queue, criterion, device, max_batches=2):
    model.eval()
    fitness_values = []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            if step >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_fitness_values = []

            for i, arch_parameters in enumerate(x_population):
                if arch_parameters.nelement() != 112:
                    reshaped = False
                    for dim_size in arch_parameters.shape:
                        if dim_size == 112:
                            arch_parameters = arch_parameters.view(-1, 112)[0]
                            reshaped = True
                            break

                    if not reshaped:
                        print(f"Error: Unable to reshape arch_parameters {i}. Current shape: {arch_parameters.size()}")
                        continue
                else:
                    arch_parameters = arch_parameters.view(112)

                reshaped_params = arch_parameters.view(14, 8)
                reshaped_params = reshaped_params.to(device)
                model.copy_arch_parameters(reshaped_params)
                logits = model(inputs)
                loss = criterion(logits, targets)
                batch_fitness_values.append(-loss.item())

            if batch_fitness_values:
                fitness_values.append(batch_fitness_values)

        if fitness_values:
            fitness_summarized = torch.tensor([sum(f) / max_batches for f in zip(*fitness_values)],
                                              dtype=torch.float32)
        else:
            print("No valid fitness values were calculated.")
            fitness_summarized = torch.tensor([], dtype=torch.float32)

    return fitness_summarized

# 引入新个体函数
def introduce_new_individuals(population, num_new_individuals):
    indices = random.sample(range(len(population.get_population())), num_new_individuals)
    for index in indices:
        new_individual = population.create_new_individual()  # 假设 Population 类有 create_new_individual 方法
        population.get_population()[index] = new_individual

# 模拟退火参数调整函数
def adjust_sa_parameters(epoch, total_epochs):
    # 随着迭代次数增加，初始温度逐渐降低
    initial_temperature = 100.0 * (1 - epoch / total_epochs)
    # 最终温度和冷却速率也可以类似地动态调整
    final_temperature = 0.1
    cooling_rate = 0.95 - 0.05 * (epoch / total_epochs)
    return initial_temperature, final_temperature, cooling_rate

# Initialize
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:{}".format(args.gpu))

torch.cuda.set_device(args.gpu)
cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

CIFAR_CLASSES = 100
criterion = nn.CrossEntropyLoss().to(device)
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, device)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)

train_transform, valid_transform = utils._data_transforms_cifar100(args)
train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))

train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=False, num_workers=2,
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))
valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    pin_memory=False, num_workers=2)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

# Create initial population
population = Population(args.pop_size, model._steps, device)

DIR = "Huber-50epoch-onestep-diffevo_search100-kde-{}".format(time.strftime("%Y%m%d-%H%M%S"))
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

logging.info("gpu device = {}".format(args.gpu))
logging.info("args =  %s", args)

writer = SummaryWriter(os.path.join(DIR, 'runs', 'de'))

# 初始化教师模型
teacher_model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, device).to(device)
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# STAGE 1
start = time.time()
for epoch in range(args.warm_up):
    start_time = time.time()
    logging.info('[INFO] Warming up!!!')
    warm_train(model, train_queue, criterion, optimizer)
    logging.info("[INFO] Warming up finished in {} minutes".format((time.time() - start_time) / 60))

# Instantiate DiffEvo
de_optimizer = DiffEvo(
    teacher_model=teacher_model,
    num_step=1,  # 修改为 1 以实现一步搜索
    density='kde',
    noise=1.0,
    lambda_kl=0.5,
    scaling=1,
    kde_bandwidth=0.1
)
# 获取使用的密度算法并输出
density_method = de_optimizer.get_density_method()
output_message = f"{density_method} was used for density estimation."

# 打印和记录
writer.add_text('Density Estimator', output_message)
print(output_message)
# STAGE 1

for epoch in range(args.epochs):
    # 模拟退火参数调整
    initial_temperature, final_temperature, cooling_rate = adjust_sa_parameters(epoch, args.epochs)
    logging.info(f"Epoch {epoch}: Initial Temperature = {initial_temperature}, Final Temperature = {final_temperature}, Cooling Rate = {cooling_rate}")

    # 引入新个体
    introduce_new_individuals(population, 5)  # 引入 5 个新个体

    logging.info("[INFO] Generation {} training with learning rate {}".format(epoch + 1, scheduler.get_lr()[0]))
    start_time = time.time()

    train(model, train_queue, criterion, optimizer, epoch + 1)
    logging.info("[INFO] Training finished in {} minutes".format((time.time()- start_time) / 60))
    torch.save(model.state_dict(), "Huber-cifar100-diffevo-onestep-kde-model.pt")
    scheduler.step()

    logging.info("[INFO] Evaluating Generation {} ".format(epoch + 1))
    validation(model, valid_queue, criterion, epoch + 1, population)

    # 获取当前种群最佳个体的验证准确率
    population.pop_sort()
    current_best_accuracy = population.get_population()[0].top1.avg

    # 评估完整验证集准确率（避免抽样偏差）
    model.eval()
    best_arch = population.get_population()[0].arch_parameters
    model.copy_arch_parameters(best_arch)
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in valid_queue:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    current_accuracy = 100 * correct / total

    # 保存最佳模型
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"[INFO] Saved best model with accuracy: {best_accuracy:.2f}%")

        # 更新教师模型为当前最佳
        teacher_model.load_state_dict(model.state_dict())
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        # 更新 DiffEvo 优化器中的教师模型
        de_optimizer.teacher_model = teacher_model

    # DiffEvo optimization
    class Individual:
        def __init__(self, arch_params):
            self.arch_parameters = arch_params
        def set_arch_parameters(self, alphas_normal, alphas_reduce):
            self.alphas_normal = alphas_normal
            self.alphas_reduce = alphas_reduce

    # 获取架构参数并转换为 NumPy 数组
    arch_parameters = population.get_arch_parameters()

    # 确保所有 arch_parameters 是 torch.Tensor 并且把它们转移到 CPU 并转换为 numpy 数组
    arch_parameters_cpu = [p.cpu().numpy() for p in arch_parameters]

    initial_population_list = []

    # 构建初始种群并确保每个 arch_parameters 被转换为一维的 NumPy 数组
    for idx, params in enumerate(arch_parameters_cpu):
        if isinstance(params, np.ndarray):
            params_flat = params.flatten()  # 使其展平为一维
        else:
            raise ValueError(f"Expected params to be numpy array, but got {type(params)}")
        initial_population_list.append(params_flat)

    initial_population_np = np.array(initial_population_list)
    x_flattened_reshaped = torch.from_numpy(initial_population_np).to(torch.float32)

    # 执行 DiffEvo 优化
    fitness_with_args = partial(fitness_function, model=model, valid_queue=valid_queue, criterion=criterion,
                                device=device)
    de_optimizer.model = model  # 添加这一行，将主模型传递给 DiffEvo

    optimized_population_objects = de_optimizer.optimize(
        x_flattened_reshaped,  # 初始种群参数（2D Tensor）
        (inputs, targets),  # 训练数据（元组）
        trace=False  # 第四个参数为 trace
    )
    # 判断返回值是否为元组
    if isinstance(optimized_population_objects, tuple):
        optimized_population = optimized_population_objects[0]
    else:
        optimized_population = optimized_population_objects

    optimized_population_np = optimized_population.cpu().numpy()
    def process_optimized_population(optimized_population_np):
        num_individuals = optimized_population_np.shape[0] // 2  # 使用前一半
        transformed_population = []
        for i in range(num_individuals):
            part_normal_all_elements = optimized_population_np[2 * i].reshape(-1)[:112]
            part_reduce_all_elements = optimized_population_np[2 * i + 1].reshape(-1)[:112]
            part_normal = part_normal_all_elements.reshape(14, 8)
            part_reduce = part_reduce_all_elements.reshape(14, 8)
            transformed_population.append([part_normal, part_reduce])
        return np.array(transformed_population)

    processed_population = process_optimized_population(optimized_population_np)
    population_instance = Population(args.pop_size, model._steps, device)
    population_instance.set_arch_parameters(processed_population)

    # 记录评价指标
    for i, p in enumerate(population.get_population()):
        writer.add_scalar("pop_top1_{}".format(i + 1), p.get_fitness(), epoch + 1)
        writer.add_scalar("pop_top5_{}".format(i + 1), p.top5.avg, epoch + 1)
        writer.add_scalar("pop_obj_valid_{}".format(i + 1), p.objs.avg, epoch + 1)

    # 保存population信息
    with open(os.path.join(DIR, "population_{}.pickle".format(epoch + 1)), 'wb') as f:
        pickle.dump(population, f)

    last = time.time() - start_time
    logging.info("[INFO] {}/{} epoch finished in {} minutes".format(epoch + 1, args.epochs, last / 60))
    utils.save(model, os.path.join(DIR, "weights", "weights.pt"))

writer.close()
last = time.time() - start
logging.info("[INFO] Total search time: {} hours".format(last / 3600))