import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import argparse

# Minimal VAE for MNIST generation
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # log_var
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    """VAE loss = Reconstruction loss + KL divergence"""
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item() / len(data):.4f}')
    
    return train_loss / len(train_loader.dataset)

def test_epoch(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var).item()
    
    return test_loss / len(test_loader.dataset)

def generate_samples(model, device, num_samples=64, output_dir='ignore-'):
    """Generate random samples from the VAE"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, 20).to(device)
        samples = model.decode(z).cpu()
        
        # Reshape to image format
        samples = samples.view(num_samples, 1, 28, 28)
        
        # Save generated images
        save_image(samples, f'{output_dir}/generated_samples.png', nrow=8, normalize=True)
        print(f'Generated samples saved to {output_dir}/generated_samples.png')

def reconstruct_images(model, test_loader, device, output_dir='ignore-'):
    """Reconstruct test images to compare with originals"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:8].to(device)  # Take first 8 images
        recon, _, _ = model(data)
        
        # Original images
        save_image(data, f'{output_dir}/original_images.png', nrow=8, normalize=True)
        
        # Reconstructed images
        recon = recon.view(-1, 1, 28, 28)
        save_image(recon, f'{output_dir}/reconstructed_images.png', nrow=8, normalize=True)
        
        print(f'Original images saved to {output_dir}/original_images.png')
        print(f'Reconstructed images saved to {output_dir}/reconstructed_images.png')

def interpolate_latent(model, test_loader, device, output_dir='ignore-', steps=10):
    """Interpolate between two images in latent space"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        img1, img2 = data[0:1].to(device), data[1:2].to(device)
        
        # Encode images to latent space
        mu1, _ = model.encode(img1.view(-1, 784))
        mu2, _ = model.encode(img2.view(-1, 784))
        
        # Interpolate in latent space
        interpolations = []
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            img_interp = model.decode(z_interp)
            interpolations.append(img_interp.view(1, 1, 28, 28))
        
        # Concatenate all interpolations
        interpolated = torch.cat(interpolations, dim=0)
        save_image(interpolated, f'{output_dir}/interpolation.png', nrow=steps, normalize=True)
        
        print(f'Latent interpolation saved to {output_dir}/interpolation.png')

def main(epochs=10, batch_size=128, learning_rate=1e-3, latent_dim=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f'VAE model created with {sum(p.numel() for p in model.parameters())/1e3:.1f}K parameters')
    print(f'Latent dimension: {latent_dim}')
    
    # Training
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = test_epoch(model, test_loader, device)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Generate samples every few epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            generate_samples(model, device, num_samples=64)
    
    # Final evaluation and generation
    print('\nGenerating final samples and reconstructions...')
    generate_samples(model, device, num_samples=64)
    reconstruct_images(model, test_loader, device)
    interpolate_latent(model, test_loader, device)
    
    print('\nTraining completed! Check the ignore- folder for generated images.')

if __name__ == '__main__':
    import debugpy
    debugpy.listen(4071)
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()

    parser = argparse.ArgumentParser(description='Minimal VAE for MNIST')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=20, help='Latent dimension')
    
    args = parser.parse_args()
    
    main(epochs=args.epochs, batch_size=args.batch_size, 
         learning_rate=args.lr, latent_dim=args.latent_dim)
