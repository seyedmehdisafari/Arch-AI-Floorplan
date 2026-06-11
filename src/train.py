import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import RPlanDataset
from src.model import UNet
import time

def train_model(data_dir, epochs=10, batch_size=16, lr=0.001):
    # 1. Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Training on Apple Silicon GPU (MPS) 🚀")
    else:
        device = torch.device("cpu")
        print("Training on CPU ⚠️")

    # 2. Data loaders
    print("Loading data...")
    dataset = RPlanDataset(data_dir=data_dir, img_size=(256, 256))
    # We set num_workers=2 to speed up data loading on your Mac's multi-core CPU
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 3. Initialize model, loss, and optimizer
    model = UNet(in_channels=1, out_channels=3).to(device)
    
    # Mean Squared Error Loss to compare pixel-by-pixel colors
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero out gradients from previous step
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass & Optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print batch progress every 20 batches
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"--> Epoch [{epoch+1}/{epochs}] Finished | Average Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s\n")
        
        # Optional: Save a checkpoint at the end of each epoch
        torch.save(model.state_dict(), "unet_rplan_latest.pth")
        
    print("Training Complete! Model saved as 'unet_rplan_latest.pth'")