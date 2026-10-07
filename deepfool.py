import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

MAX_ITER = 30
NUDGE_STEPS = 2
ETA = 0.02
overshoots = [0, .05, .10, .15, .20, .25, .30]

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
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location=device))
model.eval()

MEAN = torch.tensor([0.1307], device=device).view(1, -1, 1, 1)
STD  = torch.tensor([0.3081], device=device).view(1, -1, 1, 1)
def _normalize(x): return (x - MEAN) / STD

def deepfool_attack(image, overshoot, data_grad, max_iter=MAX_ITER, num_classes=10):
    if overshoot <= 0.0:
        return image.detach()
    x = image.detach()
    with torch.no_grad():
        out0 = model(_normalize(x))
        y0 = int(out0.argmax(dim=1).item())
    x_adv = x.clone()
    r_tot = torch.zeros_like(x)
    hit_budget = False
    if overshoot < 0.10:
        topk = num_classes - 1
    else:
        topk = 5
    steps = 0
    while steps < max_iter:
        steps += 1
        x_adv.requires_grad_(True)
        outputs = model(_normalize(x_adv))[0]
        cur = int(outputs.argmax().item())
        if cur != y0:
            break
        grad_y = torch.autograd.grad(outputs[cur], x_adv, retain_graph=True, create_graph=False)[0]
        if topk >= (num_classes - 1):
            candidate_indices = [k for k in range(num_classes) if k != cur]
        else:
            take = min(num_classes, topk + 1)
            _, idxs = torch.topk(outputs.detach(), k=take)
            candidate_indices = [int(i) for i in idxs.tolist() if int(i) != cur][:topk]
            if len(candidate_indices) < min(topk, num_classes - 1):
                rest = [k for k in range(num_classes) if k != cur and k not in candidate_indices]
                need = min(topk, num_classes - 1) - len(candidate_indices)
                candidate_indices += rest[:need]
        min_val, r_i = None, None
        for k in candidate_indices:
            grad_k = torch.autograd.grad(outputs[k], x_adv, retain_graph=True, create_graph=False)[0]
            w_k = grad_k - grad_y
            f_k = (outputs[k] - outputs[cur]).detach()
            w_norm = w_k.view(-1).norm(p=2) + 1e-12
            value = torch.abs(f_k) / w_norm
            if (min_val is None) or (value < min_val):
                min_val = value
                r_i = (torch.abs(f_k) / (w_norm ** 2)) * w_k
        if r_i is None:
            break
        factor = (1.0 + (ETA if hit_budget else 0.0))
        r_tot = r_tot + factor * r_i.detach()
        x_step = x + r_tot
        delta = torch.clamp(x_step - x, min=-float(overshoot), max=float(overshoot))
        x_next = torch.clamp(x + delta, 0.0, 1.0).detach()
        if not hit_budget and delta.abs().max() >= float(overshoot) - 1e-6:
            hit_budget = True
        elif hit_budget:
            if steps >= max_iter or (steps >= (MAX_ITER - NUDGE_STEPS)):
                x_adv = x_next
                break
        x_adv = x_next
    return x_adv.detach()

def denorm(batch, mean=[0.1307], std=[0.3081]):
    mean = MEAN if isinstance(mean, list) else mean
    std  = STD  if isinstance(std, list)  else std
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test(model, device, test_loader, overshoot):
    correct = 0
    adv_examples = []

    for idx, (data, target) in enumerate(test_loader, 1):
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

        perturbed_data = deepfool_attack(data_denorm, overshoot, data_grad)

        output = model(_normalize(perturbed_data))
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            if overshoot == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Overshoot: {overshoot}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc:.4f}")
    return final_acc, adv_examples

# -------------------- Run & Plot --------------------
accuracies, examples = [], []
for ov in overshoots:
    acc, ex = test(model, device, test_loader, ov)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(overshoots, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("DeepFool")
plt.xlabel("Overshoot")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(outdir, "accuracy_vs_overshoot_deepfool.png"), dpi=200, bbox_inches="tight")
plt.show()

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(overshoots)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(overshoots), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Overshoot: {overshoots[i]}", fontsize=14)
        orig, adv, ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "adv_examples_grid_deepfool.png"), dpi=200, bbox_inches="tight")
plt.show()
