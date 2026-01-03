from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# def set_seed(seed_value=42):
#     """Set the seed to ensure the results are reproducible."""
#     random.seed(seed_value)
#     np.random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)
    
#     # Additional configuration for the CUDNN backend (if using GPU) for absolute accuracy
#     # Note: This will slow down the training process slightly
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load MNIST dataset
# MNIST includes 28x28 grayscale handwritten digital images
# Transform: Convert to Tensor and Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalized to [-1, 1]
])

batch_size = 64

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, shuffle=False)



# 3. Define MLP model (Multi-Layer Perceptron)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Input layer: 28*28 = 784
        # Hidden Layer 1
        self.fc1 = nn.Linear(784, 512) 
        self.relu = nn.ReLU()
        # Hidden Layer 2
        self.fc2 = nn.Linear(512, 256)
        # Output Layer
        self.fc3 = nn.Linear(256, 10) 
        
    def forward(self, x):
        # Flatten [Batch_size, 1, 28, 28] -> [Batch_size, 784]
        x = x.view(-1, 28 * 28) 
        
        # Hidden Layer 1 + ReLU
        x = self.fc1(x)
        x = self.relu(x)
        
        # Hidden Layer 2 + ReLU
        x = self.fc2(x)
        x = self.relu(x)
        
        # Output Layer
        # Note: nn.CrossEntropyLoss in PyTorch includes Softmax
        # so return logits (raw scores) here.
        x = self.fc3(x)
        return x

model = SimpleMLP().to(device)

# 4.  Loss Function and Optimizer
criterion = nn.CrossEntropyLoss() # LogSoftmax + NLLLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training 
num_epochs = 5
loss_history = []

print("\nTraining started...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate epoch loss
    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 6. Loss Curve
output_folder = Path(__file__).parent / 'charts'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"File save: {output_folder}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), loss_history, marker='o', label='Training Loss')
plt.title('Lost curve across 5 epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

save_path = os.path.join(output_folder, 'loss_curve.png')
plt.savefig(save_path)
print(f"Chart saved: {save_path}")


# 7. Evaluation
print("\nEvaluating on test set...")
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on 10,000 test images: {accuracy:.2f}%')