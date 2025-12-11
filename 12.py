import torch
import torch.nn as nn
import torch.optim as optim

# Random dataset
torch.manual_seed(0)
X = torch.randn(500, 20)  # 500 samples, 20 features
y = torch.randint(0, 3, (500,))  # 3 classes
# print(X,y)
# Feed-forward neural network
class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = FFNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Accuracy
with torch.no_grad():
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == y).float().mean().item() * 100

print("Accuracy:", acc)
