import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

constants = [0, 9, 18, 27, 36, 63, 72]
# pretrained_model = "data/lenet_mnist_model.pth"
pretrained_model = "D:\pycharm project\SkyGuard-UAV-Defense-main\lenet_mnist_model.pth"
torch.manual_seed(42)
outdir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(outdir, exist_ok=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x); x = F.relu(x)
        x = self.conv2(x); x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x); x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])),
    batch_size=1, shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location=device))
model.eval()

def cw_attack(image, constant, data_grad,
              steps: int = 40,
              lr: float = 0.01,
              kappa: float = 0.0):

    x_orig = image.detach()
    c = float(constant)

    normalizer = transforms.Normalize((0.1307,), (0.3081,))

    with torch.no_grad():
        out0 = model(normalizer(x_orig))
        y0 = out0.max(1, keepdim=False)[1]

    delta = torch.zeros_like(x_orig, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    best_adv = x_orig
    best_obj = torch.full((1,), float("inf"), device=device)
    patience, patience_cnt = 8, 0

    for _ in range(steps):
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        out = model(normalizer(x_adv))

        true_logit = out.gather(1, y0.view(-1, 1)).squeeze(1)

        masked = out.clone()
        masked[torch.arange(out.size(0)), y0] = -1e10
        other_logit, _ = masked.max(dim=1)

        f = torch.clamp(true_logit - other_logit + kappa, min=0.0)
        l2 = torch.sum((x_adv - x_orig) ** 2, dim=(1, 2, 3))
        loss = l2 + c * f

        opt.zero_grad()
        loss.mean().backward()
        opt.step()

        with torch.no_grad():
            improved = loss.detach() < best_obj
            if improved.item():
                best_obj = loss.detach()
                best_adv = x_adv.detach()
                patience_cnt = 0
            else:
                patience_cnt += 1
                if f.item() == 0.0 and patience_cnt >= patience:
                    break

    return best_adv.detach()

def denorm(batch, mean=[0.1307], std=[0.3081]):
    if isinstance(mean, list): mean = torch.tensor(mean).to(device)
    if isinstance(std, list):  std  = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test(model, device, test_loader, constant):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        data_denorm = denorm(data)

        perturbed_data = cw_attack(data_denorm, constant, data_grad)

        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        output = model(perturbed_data_normalized)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if constant == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Constant(c): {constant}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    return final_acc, adv_examples


accuracies, examples = [], []
for c in constants:
    acc, ex = test(model, device, test_loader, c)
    accuracies.append(acc); examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(constants, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.title("CW")
plt.xlabel("Constant")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(outdir, "accuracy_vs_constant_cw.png"), dpi=200, bbox_inches="tight")
plt.show()

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(constants)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(constants), len(examples[0]), cnt)
        plt.xticks([], []); plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"c: {constants[i]}", fontsize=14)
        orig, adv, ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "adv_examples_grid_cw.png"), dpi=200, bbox_inches="tight")
plt.show()
