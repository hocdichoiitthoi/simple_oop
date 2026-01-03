from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# 1. Data Transformation
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. Load CIFAR-10 dataset
batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Block 1: Conv -> ReLU -> Pool
        # Input: 3 channels (RGB), Output: 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Norm
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Conv -> ReLU -> Pool
        # Input: 32 channels, Output: 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # Batch Norm

        # Block 3: Conv -> ReLU -> Pool 
        # Input: 64 channels, Output: 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # Batch Norm

        # Fully Connected Layer (Dense)
        # The image went through 3 pooling rounds: 32 -> 16 -> 8 -> 4. The final size is 4x4.
        self.flatten_dim = 128 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, self.flatten_dim)

        # Fully Connected
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 5
train_losses = []
train_accuracies = []

print("Training started...")

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Loss Statistics
        running_loss += loss.item()

        # Accuracy Statistics
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy for each epoch.
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%')

print('Training complete!')

# --- Draw charts ---

output_folder = Path(__file__).parent / 'charts'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"File save: {output_folder}")

plt.figure(figsize=(12, 5))
# Loss chart
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='red')
plt.title('Training Loss across Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy chart
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.title('Training Accuracy across Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_folder, 'training_charts.png')
plt.savefig(save_path)
print(f"Chart saved: {save_path}")
# --- Save model weights ---
PATH = Path(__file__).parent / 'model' / 'cifar10_cnn.pth'
torch.save(net.state_dict(), PATH)
print(f"Model weight saved: {PATH}")