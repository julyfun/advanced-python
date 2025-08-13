import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import argparse
import os
import math

# CIFAR-10 classification with Vision Transformer and Data Augmentation
# Vision Transformer (ViT) implementation for CIFAR-10 image classification
# Uses patch-based attention mechanism instead of convolutional layers

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)

        x = self.projection(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, n_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, n_patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (batch_size, n_heads, n_patches, head_dim)
        
        # Attention. 每一个 patch 对另一个 patch 的注意力权重是一个标量.
        # attn: (batch_size, n_heads, n_patches, n_patches)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1) # (batch_size, n_heads, n_patches, n_patches[softmax])
        attn = self.dropout(attn)
        
        # Apply attention to values
        # attn @ v: (batch_size, n_heads, n_patches, head_dim)
        #   .transpose(1, 2): (batch_size, n_patches, n_heads, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_patches, embed_dim)
        x = self.proj(x)
        return x # (batch_size, n_patches, embed_dim)

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for image classification"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=10, 
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token (batch_size, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token for classification
        return self.head(cls_token_final)

def export_to_onnx(model, onnx_path, img_size=224):
    """Export trained model to ONNX format"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'Model exported to ONNX: {onnx_path}')

def main(load_checkpoint=None, save=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomRotation(degrees=10),  # Random rotation ±10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Random crop and resize
        transforms.Resize((224, 224)),  # ViT input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])
    
    # No augmentation for test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])
    
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10('data', train=False, transform=test_transform)
    
    # [smaller dataset] - Use 3% of data for faster training
    train_size = int(0.03 * len(train_data))
    test_size = int(0.03 * len(test_data))
    train_indices = torch.randperm(len(train_data))[:train_size]
    test_indices = torch.randperm(len(test_data))[:test_size]
    train_subset = Subset(train_data, train_indices)
    test_subset = Subset(test_data, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)  # Smaller batch for ViT
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    print(f'Using {len(train_subset)} training samples ({len(train_subset)/len(train_data)*100:.1f}%) with data augmentation')
    print(f'Using {len(test_subset)} test samples ({len(test_subset)/len(test_data)*100:.1f}%) without augmentation')
    
    # [model] - Smaller ViT for CIFAR-10
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=10,  # 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        embed_dim=384,  # Smaller than standard ViT
        depth=6,        # Fewer layers
        n_heads=6,      # Fewer attention heads
        mlp_ratio=4,
        dropout=0.1
    )
    model = model.to(device)
    
    print(f'Vision Transformer model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters')
    
    # [training]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)  # AdamW for ViT
    start_epoch = 0
    
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f'Loading checkpoint from {load_checkpoint}')
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Resumed from epoch {start_epoch}')
    
    # Train for 5 epochs (ViT may need more epochs)
    for epoch in range(start_epoch, start_epoch + 5):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:  # More frequent logging for smaller dataset
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}')
    
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
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    if save:
        export_to_onnx(model, save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification with Vision Transformer')
    parser.add_argument('--load', type=str, help='Path to checkpoint to load')
    parser.add_argument('--save', type=str, default='ignore-vit-model.onnx', help='Path to export ONNX model')
    args = parser.parse_args()
    
    main(load_checkpoint=args.load, save=args.save)
