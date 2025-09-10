import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------
# 1. Define the model
# ------------------------------
class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)   # 28*28=784 inputs
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)    # 10 classes (digits)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Count parameters
model = SmallModel()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {total_params} parameters.")

# ------------------------------
# 2. Load data (MNIST digits)
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])

train_data = datasets.MNIST(root=".", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# ------------------------------
# 3. Training setup
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 4. Training loop
# ------------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ------------------------------
# 5. Evaluation
# ------------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
