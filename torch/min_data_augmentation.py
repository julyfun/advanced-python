import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Minimal MNIST classification with ResNet
def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data (no augmentation)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])
    
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Model
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for MNIST
    model = model.to(device)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 3 epochs
    for epoch in range(3):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Test
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
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()