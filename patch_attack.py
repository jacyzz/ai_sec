import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# pretrained_model = os.path.join(current_dir, "data", "lenet_mnist_model.pth")
pretrained_model = "D:\pycharm project\SkyGuard-UAV-Defense-main\lenet_mnist_model.pth"
outdir = os.path.join(current_dir, "outputs")
os.makedirs(outdir, exist_ok=True)

# 测试不同的补丁大小
patch_sizes = [6, 8, 10]  # 补丁大小从6*6到10*10
torch.manual_seed(42)


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# 只加载前300个测试样本用于快速测试
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
]))
test_subset = torch.utils.data.Subset(test_dataset, range(300))
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print(f"测试样本数量: {len(test_subset)}")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))
model.eval()


def create_patch_mask(image_size, patch_size, position='center'):
    """
    创建补丁mask
    image_size: (height, width)
    patch_size: (patch_height, patch_width)
    position: 'center' or 'random'
    """
    h, w = image_size
    ph, pw = patch_size

    mask = torch.zeros((1, h, w), device=device)

    if position == 'center':
        # 中心位置
        y_start = (h - ph) // 2
        x_start = (w - pw) // 2
    else:
        # 随机位置
        y_start = torch.randint(0, h - ph + 1, (1,)).item()
        x_start = torch.randint(0, w - pw + 1, (1,)).item()

    mask[:, y_start:y_start + ph, x_start:x_start + pw] = 1.0
    return mask, (y_start, x_start)


def generate_adversarial_patch(image, target_model, patch_size, steps=800):
    """
    使用mask生成对抗性补丁
    """
    _, h, w = image.shape
    image = image.clone().detach()

    # 创建补丁mask
    mask, patch_pos = create_patch_mask((h, w), (patch_size, patch_size), position='center')

    # 初始化补丁（随机噪声）
    patch = torch.zeros((1, h, w), device=device, requires_grad=True)

    # 创建优化器
    optimizer = optim.Adam([patch], lr=0.1)

    # 获取原始预测
    with torch.no_grad():
        output = target_model(image.unsqueeze(0))
        original_pred = output.argmax(dim=1)
        probs_clean = F.softmax(output, dim=1)
        probs_clean[0, original_pred] = 0  # 排除正确类别

    # 选择目标类别（除了正确类别外的随机类别）
    target_class = (original_pred + torch.randint(1, 10, (1,)).item()) % 10

    # 迭代优化补丁
    for step in range(steps):
        # 应用补丁：只在mask区域添加扰动
        perturbed_image = (1 - mask) * image + patch * mask
        perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)

        # 前向传播
        output = target_model(perturbed_image.unsqueeze(0))

        # 损失函数：最小化正确类别概率，最大化目标类别概率
        probs = F.softmax(output, dim=1)
        loss = probs[0, original_pred] - probs[0, target_class]

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 约束扰动大小
        with torch.no_grad():
            patch.data = torch.clamp(patch, 0, 255)
            patch.data = patch * mask  # 确保只在mask区域有扰动

    # 生成最终的对抗样本
    adversarial_image = (1 - mask) * image + patch * mask
    adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

    return adversarial_image, patch * mask


def denorm(batch, mean=[0.1307], std=[0.3081]):
    """反归一化函数"""
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test_patch_robustness(model, test_loader, patch_size):
    """
    测试模型对不同大小补丁的鲁棒性
    """
    correct = 0
    total_attacked = 0
    adv_examples = []

    print(f"\n测试补丁大小: {patch_size}x{patch_size}")
    start_time = time.time()

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # 原始预测
        with torch.no_grad():
            output = model(data)
            init_pred = output.argmax(dim=1)

        # 只攻击那些原始预测正确的样本
        if init_pred.item() == target.item():
            total_attacked += 1

            # 反归一化
            data_denorm = denorm(data)

            # 生成对抗样本
            adversarial_image, perturbation = generate_adversarial_patch(
                data_denorm.squeeze(0), model, patch_size
            )

            # 重新归一化并分类
            adversarial_normalized = transforms.Normalize((0.1307,), (0.3081,))(adversarial_image.unsqueeze(0))

            with torch.no_grad():
                output_adv = model(adversarial_normalized)
                final_pred = output_adv.argmax(dim=1)

            if final_pred.item() == target.item():
                correct += 1
            else:
                # 保存成功的攻击示例
                if len(adv_examples) < 3:
                    adv_ex = {
                        'original': data_denorm.squeeze().cpu().numpy(),
                        'adversarial': adversarial_image.squeeze().detach().cpu().numpy(),
                        'perturbation': perturbation.squeeze().detach().cpu().numpy(),
                        'original_pred': init_pred.item(),
                        'final_pred': final_pred.item(),
                        'target': target.item()
                    }
                    adv_examples.append(adv_ex)

        # 显示进度
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  进度: {i + 1}/{len(test_loader)}, 用时: {elapsed:.1f}s")

    # 计算准确率
    accuracy = correct / max(total_attacked, 1)
    attack_success_rate = 1 - accuracy if total_attacked > 0 else 0

    print(f"补丁大小: {patch_size}x{patch_size}")
    print(f"攻击样本数: {total_attacked}")
    print(f"成功防御: {correct}")
    print(f"鲁棒准确率: {accuracy:.4f}")
    print(f"攻击成功率: {attack_success_rate:.4f}")

    return accuracy, adv_examples


# 主执行流程
accuracies = []
all_examples = []
total_start_time = time.time()

print("开始测试不同大小补丁的攻击效果...")
print("=" * 50)

# 对每个补丁大小进行测试
for i, patch_size in enumerate(patch_sizes):
    patch_start_time = time.time()
    print(f"\n[{i + 1}/{len(patch_sizes)}] 测试补丁大小: {patch_size}x{patch_size}")

    acc, examples = test_patch_robustness(model, test_loader, patch_size)
    accuracies.append(acc)
    all_examples.append(examples)

    patch_time = time.time() - patch_start_time
    print(f"补丁大小 {patch_size}x{patch_size} 测试完成，用时: {patch_time:.1f}秒")

print("\n" + "=" * 50)
total_time = time.time() - total_start_time
print(f"所有测试完成！总用时: {total_time:.1f}秒")

# 绘制准确率vs补丁大小曲线
plt.figure(figsize=(10, 6))
plt.plot(patch_sizes, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs Patch size', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(patch_sizes)
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.savefig(os.path.join(outdir, "accuracy_vs_patch_size.png"), dpi=300, bbox_inches='tight')
plt.show()

# 显示对抗样本示例
plt.figure(figsize=(15, 10))
for i, patch_size in enumerate(patch_sizes):
    if all_examples[i]:  # 确保有示例可显示
        example = all_examples[i][0]  # 显示第一个示例
        plt.subplot(3, 4, i * 2 + 1)
        plt.imshow(example['original'], cmap='gray')
        plt.title(f'original ({patch_size}x{patch_size})\npred: {example["original_pred"]}')
        plt.axis('off')

        plt.subplot(3, 4, i * 2 + 2)
        plt.imshow(example['adversarial'], cmap='gray')
        plt.title(f'adv\npred: {example["final_pred"]}')
        plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(outdir, "adversarial_examples.png"), dpi=300, bbox_inches='tight')
plt.show()

print("测试结果已保存到 outputs 目录")
print(f"最终结果: 补丁大小 vs 准确率")
for patch_size, acc in zip(patch_sizes, accuracies):
    print(f"补丁 {patch_size}x{patch_size}: 准确率 = {acc:.4f}")