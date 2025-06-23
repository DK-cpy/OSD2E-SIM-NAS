from __future__ import absolute_import, division, print_function
import argparse
import archs
import datasets
from trainer.trainer_generator import GenTrainer
from trainer.trainer_utils import LinearLrDecay
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from archs.super_network import Generator, Discriminator
from archs.fully_super_network import simple_Discriminator
from algorithms.diffevo_search_algs import DiffEvoSearchAlgorithm
from diffevo import DiffEvo
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from pytorch_image_generation_metrics import get_inception_score_and_fid, get_fid, get_inception_score
import random
import math

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_selected', type=int, default=10, help='number of selected genotypes')
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type')
    parser.add_argument('--loss', type=str, default='hinge', help='loss function')
    parser.add_argument('--img_size', type=int, default=32, help='image size, 32 for cifar10, 48 for stl10')
    parser.add_argument('--bottom_width', type=int, default=4, help='init resolution, 4 for cifar10, 6 for stl10')
    parser.add_argument('--channels', type=int, default=3, help='image channels')
    parser.add_argument('--data_path', type=str, default='data/datasets/cifar-10', help='dataset path')

    parser.add_argument('--exp_name', type=str, default='new-arch_searchG_cifar10', help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default="0", help='visible GPU ids')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    
    # train
    parser.add_argument('--arch', type=str, default='arch_cifar10', help='architecture name')
    parser.add_argument('--my_max_epoch_G', type=int, default=1, help='max number of epoch for training G')
    parser.add_argument('--max_iter_G', type=int, default=None, help='max number of iteration for training G')
    parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
    parser.add_argument('--gen_bs', type=int, default=256, help='batche size of G')
    parser.add_argument('--dis_bs', type=int, default=128, help='batche size of D')
    parser.add_argument('--gf_dim', type=int, default=256, help='base channel-dim of G')
    parser.add_argument('--df_dim', type=int, default=128, help='base channel-dim of D')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--lr_decay', action='store_true', help='learning rate decay or not')
    parser.add_argument('--beta1', type=float, default=0.0, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--init_type', type=str, default='xavier_uniform',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='init type')
    parser.add_argument('--d_spectral_norm', type=str2bool, default=True,
                        help='add spectral_norm on discriminator or not')
    parser.add_argument('--g_spectral_norm', type=str2bool, default=False,
                        help='add spectral_norm on generator or not')
    parser.add_argument('--latent_dim', type=int, default=120, help='dimensionality of the latent space')

    # val
    parser.add_argument('--print_freq', type=int, default=100, help='interval between each verbose')
    parser.add_argument('--val_freq', type=int, default=5, help='interval between each validation')
    parser.add_argument('--num_eval_imgs', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--mute_max_num', type=int, default=1, help='max number of mutations per individual')
    # search
    parser.add_argument('--derived_start_epoch', type=int, default=0, help='')
    parser.add_argument('--derived_max_epoch', type=int, default=None, help='')
    parser.add_argument('--derived_epoch_interval', type=int, default=None, help='')
    parser.add_argument('--tau_max', type=float, default=5, help='max tau for gumbel softmax')
    parser.add_argument('--tau_min', type=float, default=0.1, help='min tau for gumbel softmax')
    parser.add_argument('--gumbel_softmax', type=str2bool, default=False, help='use gumbel softmax or not')
    parser.add_argument('--amending_coefficient', type=float, default=0.1, help='')
    parser.add_argument('--derive_freq', type=int, default=1, help='')
    parser.add_argument('--derive_per_epoch', type=int, default=0, help='number of derive per epoch')
    parser.add_argument('--draw_arch', type=str2bool, default=False, help='visualize the searched architecture or not')
    parser.add_argument('--early_stop', type=str2bool, default=False, help='use early stop strategy or not')
    parser.add_argument('--genotypes_exp', type=str, default='default_genotype.npy', help='ues genotypes of the experiment')
    parser.add_argument('--cr', type=float, default=0.0)
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema')
    parser.add_argument('--genotype_of_G', type=str, default='best_G.npy', help='ues genotypes of the experiment')
    parser.add_argument('--use_basemodel_D', type=str2bool, default=False, help='use the base model of D')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--base_arch', type=str, default='None', help='base arch for train')

    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--Total_evolutionary_algebra', type=int, default=2)
    parser.add_argument('--num_individual', type=int, default=20)
    parser.add_argument('--num_op_g', type=int, default=1)
    parser.add_argument('--max_model_size', type=int, default=13)

    opt = parser.parse_args()

    return opt

# 引入新个体的函数
def introduce_new_individuals(population, num_new_individuals, search_alg):
    indices = random.sample(range(len(population)), num_new_individuals)
    for index in indices:
        new_individual = search_alg.sample_fair().flatten()  # 假设这是生成新个体的方式
        new_individual_tensor = torch.FloatTensor(new_individual)  # 将 numpy.ndarray 转换为 torch.FloatTensor
        population[index] = new_individual_tensor
    return population

# 模拟退火参数调整函数
def adjust_sa_parameters(epoch, total_epochs):
    initial_temperature = 100.0 * (1 - epoch / total_epochs)
    final_temperature = 0.1
    cooling_rate = 0.95 - 0.05 * (epoch / total_epochs)
    return initial_temperature, final_temperature, cooling_rate

def main():
    # 计算激活统计量的函数（从项目中提取或实现）
    def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
        """计算激活值的均值和协方差矩阵"""
        # 假设项目中有类似的实现
        # 这里简化处理，实际实现可能需要调整
        from utils.inception_score import get_activations
        act = get_activations(images, sess, batch_size, verbose)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    # 计算 Frechet 距离的函数
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """计算两个分布之间的 Frechet 距离"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
        
        diff = mu1 - mu2
        
        # 计算 sqrt(sigma1 dot sigma2)
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 数值误差处理
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    
    # 重新设计的适应度评估函数
    def evaluate_fitness(trainer, genotype, fid_stat):
        # 保存当前网络状态
        original_state_dict = deepcopy(trainer.gen_net.state_dict())
        
        # 应用新的基因型
        trainer.search_alg.apply_genotype(trainer.gen_net, genotype)
        
        # 准备生成样本
        model = trainer.gen_net
        device = next(model.parameters()).device
        z_dim = model.module.z_dim if hasattr(model, 'module') else model.z_dim
        
        # 生成样本
        with torch.no_grad():
            z = torch.randn(100, z_dim).to(device)  # 生成100个样本
            generated_images = model(z).cpu().numpy()  # 转换为 numpy 格式
        
        # 预处理：确保图像值在 [0, 255] 且维度正确 (NHWC)
        if generated_images.shape[1] in [1, 3]:  # 如果是 NCHW 格式
            generated_images = generated_images.transpose(0, 2, 3, 1)  # 转换为 NHWC
        generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)
        
        # 加载真实数据的统计量
        fid_data = np.load(fid_stat)
        mu_real = fid_data['mu']
        sigma_real = fid_data['sigma']
        
        # 计算生成图像的激活统计量
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mu_gen, sigma_gen = calculate_activation_statistics(generated_images, sess)
        
        # 计算 FID
        fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        
        # 计算 IS
        is_mean, _ = get_inception_score(generated_images)
        
        # 组合适应度：IS 越高越好，FID 越低越好
        fitness = is_mean - fid
        
        # 恢复原始网络状态
        trainer.gen_net.load_state_dict(original_state_dict)
        
        return fitness
    
    args = parse_args()
    torch.cuda.manual_seed(args.random_seed)
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
        args.gpu_ids = args.gpu_ids[1:]
    else:
        args.gpu_ids = args.gpu_ids

    # genotype G
    search_alg = DiffEvoSearchAlgorithm(args)

    # import network from genotype
    basemodel_gen = Generator(args)
    gen_net = torch.nn.DataParallel(
        basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    basemodel_dis = simple_Discriminator()
    dis_net = torch.nn.DataParallel(
        basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    max_epoch_for_D = args.my_max_epoch_G * args.n_critic
    args.max_epoch_D = args.my_max_epoch_G * args.n_critic
    max_iter_D = max_epoch_for_D * len(train_loader)

    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    start_epoch = 0
    best_fid = 1e4

    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))

    genotype_fixG = np.load(os.path.join('exps', 'best_G.npy'))
    trainer_gen = GenTrainer(args, gen_net, dis_net, gen_optimizer,
                             dis_optimizer, train_loader, search_alg, None,
                             genotype_fixG)

    best_genotypes = None
    is_mean_best = 0.0
    fid_mean_best = 999.0

    temp_model_size = args.max_model_size
    args.max_model_size = 999

    epoch_record = []
    is_record = []
    fid_record = []

    for epoch in tqdm(range(int(start_epoch), int(2)), desc='training the supernet_G:'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        trainer_gen.train(epoch, writer_dict, fid_stat, lr_schedulers)
        if epoch == 2:
            break
        if epoch >= 9999 and epoch % 5 == 0:
            now_is_max = 0
            now_fid_min = 999
            trainer_gen.clear_bag()
            # 生成初始种群：每个基因型展平为21维，堆叠为 (12, 21)
            population = np.stack(
                [search_alg.sample_fair().flatten() for i in range(12)],
                axis=0
            )  # 形状：(12, 21)

            optimizer = DiffEvo(
                teacher_model=gen_net,
                num_step=1,
                density='kde',
                noise=1.0,
                lambda_kl=0.3
            )

            train_data = next(iter(train_loader))
            # 优化器应返回 (12, 21) 的二维张量
            result = optimizer.optimize(population, train_data, gen_net, fid_stat)
            # 检查返回值是否为元组，如果是则提取第一个元素
            if isinstance(result, tuple):
                population = result[0]
            else:
                population = result

            # 确保种群维度正确：(n_samples, 21)
            assert population.ndim == 2, f"Population must be 2D, got {population.ndim}D"
            assert population.shape[1] == 21, f"Expected 21 features per genotype, got {population.shape[1]}"

            # 引入新个体
            population = introduce_new_individuals(population, num_new_individuals=3, search_alg=search_alg)

            for kk in tqdm(range(4), desc='Evaluating of subnet performance using evolutionary algorithms'):
                # 将每个21维向量恢复为 (3, 7)
                reshaped_population = [g.reshape(3, 7) for g in population]
                # 传入三维基因型矩阵进行评估
                population, pop_selected, is_mean, fid_mean, is_max, fid_min = trainer_gen.my_search_evol(
                    reshaped_population, fid_stat, kk
                )
                # 评估后重新展平为 (n_samples, 21)
                population = np.stack([g.flatten() for g in population], axis=0)
                if is_max > now_is_max:
                    now_is_max = is_max
                if fid_min < now_fid_min:
                    now_fid_min = fid_min
            epoch_record.append(epoch)
            is_record.append(now_is_max)
            fid_record.append(now_fid_min)
            np.save('epoch_record_518.npy', np.array(epoch_record))
            np.save('is_record_518.npy', np.array(is_record))
            np.save('fid_record_518.npy', np.array(fid_record))

            trainer_gen.clear_bag()

            if is_mean > is_mean_best:
                is_mean_best = is_mean
                checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_best_is.pt')
                ckpt = {'epoch': epoch,
                        'weight_G': gen_net.state_dict(),
                        'weight_D': dis_net.state_dict(),
                        'up_G_fixed': search_alg.Up_G_fixed,
                        'normal_G_fixed': search_alg.Normal_G_fixed,
                        }
                torch.save(ckpt, checkpoint_file)
                del ckpt
            if fid_mean < fid_mean_best:
                fid_mean_best = fid_mean
                checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_best_fid.pt')
                ckpt = {'epoch': epoch,
                        'weight_G': gen_net.state_dict(),
                        'weight_D': dis_net.state_dict(),
                        'up_G_fixed': search_alg.Up_G_fixed,
                        'normal_G_fixed': search_alg.Normal_G_fixed,
                        }
                torch.save(ckpt, checkpoint_file)
                del ckpt

        if epoch == args.warmup * args.n_critic:
            checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'gen_checkpoint_before_prune.pt')
            ckpt = {'epoch': epoch,
                    'weight_G': gen_net.state_dict(),
                    'weight_D': dis_net.state_dict(),
                    'up_G_fixed': search_alg.Up_G_fixed,
                    'normal_G_fixed': search_alg.Normal_G_fixed,
                    }
            torch.save(ckpt, checkpoint_file)
            del ckpt
            trainer_gen.directly_modify_fixed(fid_stat)
            logger.info(
                f'search_alg.Normal_G_fixed: {search_alg.Normal_G_fixed}, search_alg.Up_G_fixed: {search_alg.Up_G_fixed},@ epoch {epoch}.')

    checkpoint_file = os.path.join(args.path_helper['ckpt_path'], 'supernet_gen.pt')
    ckpt = {
        'weight_G': gen_net.state_dict(),
        'up_G_fixed': search_alg.Up_G_fixed,
        'normal_G_fixed': search_alg.Normal_G_fixed,
    }
    torch.save(ckpt, checkpoint_file)
    args.max_model_size = temp_model_size

    # 最终搜索阶段：同样确保种群维度正确
    population = np.stack(
        [search_alg.sample_fair().flatten() for i in range(args.num_individual)],
        axis=0
    )  # 形状：(num_individual, 21)

    optimizer = DiffEvo(
        teacher_model=gen_net,
        num_step=1,
        density='kde',
        noise=1.0,
        lambda_kl=0.3
    )
    train_data = next(iter(train_loader))

    # 模拟退火参数
    initial_temperature, final_temperature, cooling_rate = adjust_sa_parameters(0, args.Total_evolutionary_algebra)

    for ii in tqdm(range(args.Total_evolutionary_algebra), desc='search genearator using evo alg'):
        result = optimizer.optimize(population, train_data, gen_net, fid_stat)
        if isinstance(result, tuple):
            population = result[0]
        else:
            population = result
        # 验证维度
        assert population.ndim == 2 and population.shape[1] == 21, "Invalid genotype dimension"

        # 转换为 numpy 数组并 reshape
        reshaped_population = []
        for g in population:
            # 处理可能的 torch 张量（转换为 numpy 数组）
            g_np = g.numpy() if isinstance(g, torch.Tensor) else g
            reshaped = g_np.reshape(3, 7)
            assert reshaped.shape == (3, 7), f"Genotype shape error: {reshaped.shape}, expected (3, 7)"
            reshaped_population.append(reshaped)

        # 引入新个体
        population = introduce_new_individuals(population, num_new_individuals=3, search_alg=search_alg)

        # 传入形状正确的 numpy 数组列表
        new_population, pop_selected, is_mean, fid_mean, _, _ = trainer_gen.my_search_evolv2(
            reshaped_population, fid_stat, ii
        )
        # 重新展平为 21 维向量
        new_population = np.stack([g.flatten() for g in new_population], axis=0)

        # 计算当前种群和优化后种群的适应度
        current_fitness = [evaluate_fitness(trainer_gen, g.numpy().reshape(3, 7), fid_stat) for g in population]
        new_fitness = [evaluate_fitness(trainer_gen, g.numpy().reshape(3, 7), fid_stat) for g in new_population]

        # 模拟退火算法
        temperature = initial_temperature
        while temperature > final_temperature:
            for i in range(args.num_individual):
                delta_fitness = new_fitness[i] - current_fitness[i]
                if delta_fitness > 0 or random.random() < math.exp(delta_fitness / temperature):
                    # 接受新的个体
                    population[i] = new_population[i]
                    current_fitness[i] = new_fitness[i]
            temperature *= cooling_rate

    for index, geno in enumerate(pop_selected):
        file_path = os.path.join(args.path_helper['ckpt_path'],
                                 "best_gen_{}.npy".format(str(index)))
        np.save(file_path, geno)

if __name__ == '__main__':
    main()