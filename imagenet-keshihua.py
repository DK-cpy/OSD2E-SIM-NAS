import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取日志文件
log_file = './checkpoints/eval-try-20241214-085839/log.txt'

# 用于存储提取的数据
data = {
    'step': [],
    'objs': [],
    'R1': [],
    'R5': [],
    'train_acc': [],
    'valid_acc_top1': [],
    'valid_acc_top5': []
}

# 正则表达式匹配
train_pattern = re.compile(r'TRAIN Step:\s*(\d+).*Objs:\s*([\d.e+-]+).*R1:\s*([\d.e+-]+).*R5:\s*([\d.e+-]+)')
valid_pattern = re.compile(r'VALID.*Step:\s*\d+.*R1:\s*([\d.e+-]+).*R5:\s*([\d.e+-]+)')
train_acc_pattern = re.compile(r'Train_acc:\s*([\d.e+-]+)')
valid_acc_pattern = re.compile(r'Valid_acc_top1:\s*([\d.e+-]+)')
epoch_pattern = re.compile(r'Epoch:\s*(\d+).*')

# 逐行读取文件
with open(log_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    # 查找训练步骤
    train_match = train_pattern.search(line)
    if train_match:
        step = int(train_match.group(1))
        objs = float(train_match.group(2))
        R1 = float(train_match.group(3))
        R5 = float(train_match.group(4))

        data['step'].append(step)
        data['objs'].append(objs)
        data['R1'].append(R1)
        data['R5'].append(R5)
        # 补充训练准确率和验证准确率的默认值
        data['train_acc'].append(None)  # 或者 0
        data['valid_acc_top1'].append(None)  # 或者 0
        data['valid_acc_top5'].append(None)  # 或者 0
        continue

    # 查找验证准确率
    valid_match = valid_pattern.search(line)
    if valid_match:
        R1 = float(valid_match.group(1))
        R5 = float(valid_match.group(2))

        # 确保对应的 train_acc 也有值
        if len(data['train_acc']) > 0:
            data['train_acc'][-1] = None  # 更新最近的训练准确率，保持长度一致

        data['valid_acc_top1'].append(R1)
        data['valid_acc_top5'].append(R5)
        continue

    # 查找训练准确率
    train_acc_match = train_acc_pattern.search(line)
    if train_acc_match:
        data['train_acc'].append(float(train_acc_match.group(1)))
        continue

    # 查找当前epoch
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        continue

# 处理长度不一致的问题
max_length = max(len(data['step']), len(data['objs']), len(data['R1']),
                 len(data['R5']), len(data['train_acc']),
                 len(data['valid_acc_top1']), len(data['valid_acc_top5']))

# 填充缺失值
for key in data.keys():
    while len(data[key]) < max_length:
        data[key].append(None)  # 或者使用 np.nan

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 设置绘图风格
sns.set(style="whitegrid", palette="muted")

# 创建图形和坐标轴，增大图形尺寸
fig, ax1 = plt.subplots(figsize=(14, 8))  # 增加图形的宽度和高度

# 绘制训练和验证的准确率
sns.lineplot(data=df, x='step', y='R1', label='Train R1 (Top-1 Accuracy)', ax=ax1,
             color='#ff7f0e', linestyle='-', marker='o', markersize=5)  # 蓝色实线
sns.lineplot(data=df, x='step', y='valid_acc_top1', label='Valid R1 (Top-1 Accuracy)', ax=ax1,
             color='#2ca02c', linestyle='-', marker='o', markersize=5)  # 绿色实线

ax1.set_ylabel('Accuracy (%)')
ax1.set_xlabel('Training Steps')
ax1.set_title('Training Progress', fontsize=16)
ax1.legend(loc='upper left')

# 创建第二个坐标轴
ax2 = ax1.twinx()
# 使用不同的颜色和三角形标记绘制验证的 Top-5 准确率
sns.lineplot(data=df, x='step', y='valid_acc_top5', label='Valid R5 (Top-5 Accuracy)', ax=ax2,
             color='#ff7f0e', linestyle='--', marker='^', markersize=8)  # 橙色虚线，三角形标记
ax2.set_ylabel('Top-5 Accuracy (%)')
ax2.legend(loc='upper right')

# 添加解释文本，放置在图形内部
plt.figtext(0.5, 0.02,  # 调整为内部位置
            "Figure Explanation:\n"
            "This figure shows the training progress of the model. "
            "The blue line represents the training Top-1 accuracy (R1) as a function of training steps, "
            "while the green line represents the validation Top-1 accuracy (valid R1). "
            "The orange dashed line with triangle markers shows the validation Top-5 accuracy (valid R5). "
            "The left y-axis corresponds to accuracy percentages, "
            "and the right y-axis corresponds to the Top-5 accuracy value.",
            wrap=True, horizontalalignment='center', fontsize=12)

# 调整底部边距
plt.subplots_adjust(bottom=0.2)  # 增加底部边距

# 显示图形
plt.show()
