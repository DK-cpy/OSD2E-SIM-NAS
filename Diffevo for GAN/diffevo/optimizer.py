from .ddim import DDIMScheduler
from .generator import SIMGenerator
from .fitnessmapping import Identity
import torch
from tqdm import tqdm
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from utils.fid_score import calculate_activation_statistics, calculate_frechet_distance
import tensorflow as tf

class DiffEvo:
    def __init__(self,
                 teacher_model,
                 num_step: int = 1,
                 density='kde',
                 noise: float = 1.0,
                 lambda_kl: float = 0.5,
                 scaling: float = 1,
                 fitness_mapping=None,
                 kde_bandwidth=0.1):
        self.num_step = num_step
        self.lambda_kl = lambda_kl
        self.teacher_model = teacher_model

        if not density in ['uniform', 'kde', 'gmm']:
            raise NotImplementedError(f'Density estimator {density} is not implemented.')
        self.density = density
        self.kde_bandwidth = kde_bandwidth
        self.scaling = scaling
        self.noise = noise
        if fitness_mapping is None:
            self.fitness_mapping = Identity()
        else:
            self.fitness_mapping = fitness_mapping
        self.scheduler = DDIMScheduler(self.num_step)

    def optimize(self, initial_population, train_data, model, trace=False):
        x = torch.tensor(initial_population, dtype=torch.float32)
        if x.ndim != 2:
            raise ValueError(f"initial_population should be a 2D tensor, but got shape {x.shape}")
        x = x.to(train_data[0].device)
        fitness_count = []
        population_trace = [x] if trace else []

        for t, alpha in tqdm(self.scheduler):
            generator = SIMGenerator(
                x,
                fitness=self._compute_fitness(x, model, train_data),
                alpha=alpha,
                teacher_model=self.teacher_model,
                lambda_kl=self.lambda_kl
            )
            generator.model = model

            x_next = generator.generate(
                inputs=train_data[0],
                targets=train_data[1],
                noise=self.noise
            )

            if trace:
                population_trace.append(x_next.clone())
            x = x_next

        if trace:
            return x, population_trace, fitness_count
        else:
            return x

    def _compute_fitness(self, arch_params, model, train_data, fid_stat):
        model.copy_arch_parameters(arch_params)
        # 生成样本（假设 model 是生成器，输入为随机噪声）
        with torch.no_grad():
            z = torch.randn(100, model.z_dim).to(train_data[0].device)  # 假设生成100个样本
            generated_images = model(z).cpu().numpy()  # 转换为 numpy 格式（HWC，0-255）
        
        # 预处理：确保图像值在 [0, 255] 且维度正确（NHWC）
        generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)
        
        # 加载真实数据的统计量（fid_stat 是 .npz 文件路径，包含 mu_real 和 sigma_real）
        fid_data = np.load(fid_stat)
        mu_real = fid_data['mu']
        sigma_real = fid_data['sigma']
        
        # 计算生成图像的激活统计量（需要 TensorFlow 会话）
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mu_gen, sigma_gen = calculate_activation_statistics(generated_images, sess)
        
        # 计算 FID
        fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        
        # 计算 IS（假设另外有函数计算 IS，需自行实现或导入）
        # 这里假设你有一个 get_inception_score 函数，参数为生成图像
        is_mean, _ = get_inception_score(generated_images)  # 需根据实际情况实现
        
        # 组合适应度：IS 越高越好，FID 越低越好
        fitness = is_mean - fid
        return fitness