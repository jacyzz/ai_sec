import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]

pretrained_model = "D:\pycharm project\SkyGuard-UAV-Defense-main\lenet_mnist_model.pth"
# Set random seed for reproducibility
torch.manual_seed(42)
outdir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(outdir, exist_ok=True)

# LeNet Model definition
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
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# # MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location=device))
model.eval()

def pgd_attack(image, epsilon, data_grad, steps: int = 10, alpha: float = None, random_start: bool = True):

    if epsilon == 0 or epsilon <= 0.0:
        return image.detach()
    if alpha is None:
        alpha = float(epsilon) / float(max(steps, 1))
    x_orig = image.detach()
    x_adv = x_orig.clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    normalizer = transforms.Normalize((0.1307,), (0.3081,))

    with torch.no_grad():
        out0 = model(normalizer(x_orig))
        y0 = out0.max(1, keepdim=False)[1]
    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad()
        out = model(normalizer(x_adv))
        loss = F.nll_loss(out, y0)
        loss.backward()
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv + alpha * grad_sign
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv

def denorm(batch, mean=[0.1307], std=[0.3081]):

    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect ``datagrad``
        data_grad = data.grad.data
        # Restore the data to its original scale
        data_denorm = denorm(data)
        perturbed_data = pgd_attack(data_denorm, epsilon, data_grad)
        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon (PGD)")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(outdir, f"accuracy_vs_epsilon_pgd.png"), dpi=200, bbox_inches="tight")
plt.show()

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"adv_examples_grid_pgd.png"), dpi=200, bbox_inches="tight")
plt.show()
