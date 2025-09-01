import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

def train_model(model, train_loader, epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Load checkpoint path')
    parser.add_argument('--save', type=str, help='Save checkpoint path')
    parser.add_argument('--train', type=int, help='Number of training epochs')
    parser.add_argument('--eval', action='store_true', help='Evaluate on test set')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    model = SimpleNet().to(device)
    
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print(f'Loaded model from {args.load}')
    
    if args.train:
        train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        train_model(model, train_loader, args.train, device)
    
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f'Model saved to {args.save}')
    
    if args.eval:
        test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()