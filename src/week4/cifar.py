from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data Transformation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

# 2. Load CIFAR-10 dataset
batch_size = 32

full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

train_size = int(0.8 * len(full_trainset)) 
val_size = len(full_trainset) - train_size 

train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

print(f"Data after split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(testset)}")

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#         'dog', 'frog', 'horse', 'ship', 'truck')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_dim = 128 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flatten_dim)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=5e-4
)

num_epochs = 10
train_losses = []
train_accuracies = []
val_losses = []     
val_accuracies = []  

print("Training started...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    model.eval() 
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad(): 
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(valloader)
    val_epoch_acc = 100 * val_correct / val_total
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    print(f'Epoch [{epoch + 1}/{num_epochs}] '
          f'| Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% '
          f'| Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%')
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
plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='--') 
plt.title('Loss: Train vs Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy chart
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Acc', color='blue')
plt.plot(val_accuracies, label='Validation Acc', color='cyan', linestyle='--') 
plt.title('Accuracy: Train vs Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

save_path = os.path.join(output_folder, 'training_charts.png')
plt.savefig(save_path)
print(f"Chart saved: {save_path}")

# --- Save model weights ---
PATH = Path(__file__).parent / 'model' / 'cifar10_cnn.pth'
torch.save(model.state_dict(), PATH)
print(f"Model weight saved: {PATH}")

print("\nEvaluating on Test set...")
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print(f'Final Test Accuracy: {100 * test_correct / test_total:.2f}%')