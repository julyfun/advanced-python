# accelerate launch min_vit_accelerate.py --train 1 --eval 
# Epoch 1/1, Loss: 1.8571
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import math
from tqdm import tqdm
from accelerate import Accelerator

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=192, n_heads=3):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=192, n_heads=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, n_classes=10, embed_dim=192, depth=6, n_heads=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])

def train_model(model, train_loader, epochs, accelerator):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    model, optimizer, train_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, criterion
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False) as tepoch:
            for batch_idx, (data, target) in enumerate(tepoch):
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
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    model = VisionTransformer()
    
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location='cpu'))
        if accelerator.is_main_process:
            print(f'Loaded model from {args.load}')
    
    if args.train:
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        train_model(model, train_loader, args.train, accelerator)
    
    if args.save:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            torch.save(unwrapped_model.state_dict(), args.save)
            print(f'Model saved to {args.save}')
    
    if args.eval:
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        evaluate_model(model, test_loader, accelerator)

if __name__ == '__main__':
    main()
