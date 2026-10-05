import re

import matplotlib.pyplot as plt

# 读取文件
with open('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/sem_seg/2023-11-22_13-45/logs/Unet_sem_seg.txt', 'r') as f:
    lines = f.readlines()

# 初始化列表以保存损失和分数
losses = []
val_scores = []
dice_scores = []

# 初始化计数器
count = 0

# 遍历每一行
for line in lines:
    # 检查是否包含损失信息
    if 'loss =' in line:
        # 使用正则表达式提取损失值
        loss = re.search('loss = ([\d\.]+)', line)
        if loss:
            losses.append(float(loss.group(1)))
            count += 1

    # 检查是否包含验证分数信息
    if 'val_score =' in line:
        # 使用正则表达式提取验证分数值
        val_score = re.search('val_score = ([\d\.]+)', line)
        if val_score:
            # 重复最新的验证分数和Dice分数以匹配损失的数量
            for _ in range(count):
                val_scores.append(float(val_score.group(1)))

    # 检查是否包含Dice分数信息
    if 'Validation Dice score:' in line:
        # 使用正则表达式提取Dice分数值
        dice_score = re.search('Validation Dice score: ([\d\.]+)', line)
        if dice_score:
            # 重复最新的验证分数和Dice分数以匹配损失的数量
            for _ in range(count):
                dice_scores.append(float(dice_score.group(1)))
            # 重置计数器
            count = 0



# 从第20000个epoch开始
start_epoch = 125000  # 20000
epochs = list(range(start_epoch, start_epoch + len(losses[start_epoch:])))
losses = losses[start_epoch:]
val_scores = val_scores[start_epoch:]
dice_scores = dice_scores[start_epoch:]


# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Train Loss')
plt.plot(val_scores, label='Validation Loss')
plt.plot(dice_scores, label='Dice Score')
plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()