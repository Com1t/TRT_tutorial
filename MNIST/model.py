# This file contains functions for training a PyTorch MNIST Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import os
import time
from random import randint

# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class MnistModel(object):
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 100
        self.learning_rate = 0.001
        self.log_interval = 100
        
        # GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Device State:', self.device)

        # Fetch MNIST data set.
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "/tmp/mnist/data",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600,
        )
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "/tmp/mnist/data",
                train=False,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            ),
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=1,
            timeout=600,
        )
        self.network = Net().to(self.device)

    # Train the network for a single epoch
    def train(self, epoch):
        self.network.train()
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        for batch, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            output = self.network(data)

            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            if batch % self.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch / len(self.train_loader),
                        loss.data.item(),
                    )
                )

    # Test the network
    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            with torch.no_grad():
                data, target = data.to(self.device), target.to(self.device)
            output = self.network(data)
            test_loss += F.cross_entropy(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(self.test_loader)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(self.test_loader.dataset), 100.0 * correct / len(self.test_loader.dataset)
            )
        )

    # benchmark
    def benchmark(self):
        self.network.eval()
        correct = 0
        torch_total_time = 0
        for data, target in self.test_loader:
            torch_start = time.time_ns()
            with torch.no_grad():
                data, target = data.to(self.device), target.to(self.device)
            output = self.network(data)
            torch_total_time += time.time_ns() - torch_start
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        print(
            "\nTest set: Accuracy: {}/{} ({:.0f}%). Time: {:.4f} ms\n".format(
                correct, len(self.test_loader.dataset), 100.0 * correct / len(self.test_loader.dataset), torch_total_time / 10e6
            )
        )

    # Train the network for one or more epochs, validating after each epoch.
    def learn(self, num_epochs=2):
        for e in range(num_epochs):
            self.train(e + 1)
            self.test()

    # load model weights
    def load(self):
        weights = torch.load('mnist.pt')
        self.network.load_state_dict(weights)

    # save model weights
    def save(self):
        torch.save(self.network.state_dict(), 'mnist.pt')

    def get_weights(self):
        return self.network.state_dict()


def main():
    # Train the PyTorch model
    mnist_model = MnistModel()

    if os.path.exists('mnist.pt'):
        print("Found pretrained weight!")
        mnist_model.load()
        mnist_model.benchmark()
    else:
        print("No pretrained weight! Train from scratch!")
        mnist_model.learn()
        mnist_model.save()

if __name__ == "__main__":
    main()
