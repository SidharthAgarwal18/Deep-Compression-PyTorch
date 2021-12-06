import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet, LeNet_5
from net.quantization import apply_weight_sharing
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--wmFtune',type = bool,default  = False,help = "If you want to fine tune on wms")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 4} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                    transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
model = LeNet_5(mask=True).to(device)

print(model)
util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()


wminputs = []
wmtargets = []

if args.wmFtune:
    watermark = torch.load("watermark.pth",map_location = device)
    
    total_dims = len(watermark["inner_img"])
    samples_per_dim = watermark["inner_img"][0].shape[0]
    zeros = torch.zeros(1,1,64,64)

    for dim in total_dims:
        for sample in samples_per_dim:
            if not torch.equal(zeros,watermark["inner_img"][dim][sample]):
                wminputs.append((watermark["inner_img"][dim][sample]).to(device))
                wmtargets.append((watermark["inner_pred"][dim][sample]).to(device))

            if not torch.equal(zeros,watermark["outer_img"][dim][sample]):
                wminputs.append((watermark["outer_img"][dim][sample]).to(device))
                wmtargets.append((watermark["outer_pred"][dim][sample]).to(device))
    print("Total watermarks fine tuned on are : "+str(len(wminputs)))


def train(epochs,wmFineTune = False,wminputs = [],wmtargets = []):
    model.train()
    for epoch in range(epochs):

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        wm_idx = np.random.randint(len(wminputs))

        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)

            if wmFineTune:
                data = torch.cat([data, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
                target = torch.cat([target, wmtargets[(wm_idx + batch_idx) % len(wmtargets)]], dim=0)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# Initial training
print("--- Initial training ---")
train(args.epochs)
accuracy = test()
print(f"initial_accuracy {accuracy}")
torch.save(model.state_dict(), f"saves/net2.pth")
print("--- Before pruning ---")
util.print_nonzeros(model)

# Pruning
model.prune_by_std(args.sensitivity)
accuracy = test()
print(f"accuracy_after_pruning {accuracy}")
print("--- After pruning ---")
util.print_nonzeros(model)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(args.epochs,args.wmFtune,wminputs,wmtargets)
torch.save(model.state_dict(), f"saves/model_compressed.pth")
accuracy = test()
print(f"accuracy_after_retraining {accuracy}")

print("--- After Retraining ---")
util.print_nonzeros(model)
