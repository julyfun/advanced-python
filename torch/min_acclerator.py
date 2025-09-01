import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from accelerate import Accelerator

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

def train_model(model, train_loader, epochs, accelerator):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    model, optimizer, train_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, criterion
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
        
        if accelerator.is_main_process:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, accelerator):
    model.eval()
    correct = 0
    total = 0
    
    test_loader = accelerator.prepare(test_loader)
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    correct = accelerator.gather_for_metrics(torch.tensor(correct, device=accelerator.device))
    total = accelerator.gather_for_metrics(torch.tensor(total, device=accelerator.device))
    
    if accelerator.is_main_process:
        accuracy = 100 * correct.sum().item() / total.sum().item()
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Load checkpoint path')
    parser.add_argument('--save', type=str, help='Save checkpoint path')
    parser.add_argument('--train', type=int, help='Number of training epochs')
    parser.add_argument('--eval', action='store_true', help='Evaluate on test set')
    args = parser.parse_args()
    
    accelerator = Accelerator()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    model = SimpleNet()
    
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location='cpu'))
        if accelerator.is_main_process:
            print(f'Loaded model from {args.load}')
    
    if args.train:
        train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        train_model(model, train_loader, args.train, accelerator)
    
    if args.save:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            torch.save(unwrapped_model.state_dict(), args.save)
            print(f'Model saved to {args.save}')
    
    if args.eval:
        test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        evaluate_model(model, test_loader, accelerator)

if __name__ == '__main__':
    main()
