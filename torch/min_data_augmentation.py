import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import argparse
import os

def main(load_checkpoint=None, save_checkpoint='checkpoint.ckpt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),  # Random rotation ±10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Random crop and resize
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])
    
    # No augmentation for test data
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])
    
    train_data = datasets.FashionMNIST('data', train=True, download=True, transform=train_transform)
    test_data = datasets.FashionMNIST('data', train=False, transform=test_transform)
    
    # [smaller dataset]
    train_size = int(0.03 * len(train_data))
    test_size = int(0.03 * len(test_data))
    train_indices = torch.randperm(len(train_data))[:train_size]
    test_indices = torch.randperm(len(test_data))[:test_size]
    train_subset = Subset(train_data, train_indices)
    test_subset = Subset(test_data, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    print(f'Using {len(train_subset)} training samples ({len(train_subset)/len(train_data)*100:.1f}%) with data augmentation')
    print(f'Using {len(test_subset)} test samples ({len(test_subset)/len(test_data)*100:.1f}%) without augmentation')
    
    # [model]
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for MNIST
    model = model.to(device)
    
    # [training]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_epoch = 0
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f'Loading checkpoint from {load_checkpoint}')
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Resumed from epoch {start_epoch}')
    
    for epoch in range(start_epoch, start_epoch + 3):
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
    
    # [test]
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
    
    # [save]
    if save_checkpoint:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': 100 * correct / total
        }
        torch.save(checkpoint, save_checkpoint)
        print(f'Checkpoint saved to {save_checkpoint}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Classification with ResNet')
    parser.add_argument('--load', type=str, help='Path to checkpoint to load')
    parser.add_argument('--save', type=str, default='ignore-checkpoint.ckpt', help='Path to save checkpoint')
    args = parser.parse_args()
    
    main(load_checkpoint=args.load, save_checkpoint=args.save)
