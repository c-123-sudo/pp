import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], device=device)

model = nn.Linear(in_features=1, out_features=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
# Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)
    # Backward + optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print occasionally
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
# Test the trained model
with torch.no_grad():
    test_x = torch.tensor([[5.0]], device=device)
    predicted = model(test_x)
    print("Prediction for x=5:", predicted.item())
