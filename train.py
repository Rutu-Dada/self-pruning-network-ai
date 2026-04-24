import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import PrunableNet
from utils import compute_sparsity, plot_gates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Better preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


def train_model(lambda_sparse):
    model = PrunableNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            gates = model.get_all_gates()

            # 🔥 STRONG sparsity push
            sparsity_loss = torch.mean(gates) * 10

            total_loss = loss + lambda_sparse * sparsity_loss
            total_loss.backward()

            optimizer.step()

        print(f"Epoch {epoch+1} done")

    return model


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    # 🔥 Proper lambda range
    lambdas = [0.001, 0.1, 1.0]

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")

        model = train_model(lam)
        acc = evaluate(model)

        gates = model.get_all_gates()
        sparsity = compute_sparsity(gates)

        print(f"Accuracy: {acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        plot_gates(gates, name=f"lambda_{lam}")