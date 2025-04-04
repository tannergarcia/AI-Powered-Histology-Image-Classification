import torch
from torch.utils.data import DataLoader
from src.data_processing.patch_dataset import PatchDataset
from src.data_processing.simclr_transforms import SimCLRTransform
import torch.optim as optim
from src.models.simclr_model import SimCLRModel
from src.models.nt_xent_loss import NTXentLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset and dataloader
transform = SimCLRTransform(size=224)
dataset = PatchDataset(root_dir='data/patches/BCC', transform=transform)
batch_size = 128  # Adjust based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Initialize the SimCLR model
model = SimCLRModel(base_model='resnet18', out_dim=128)
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = NTXentLoss(temperature=0.5)

# Training loop with mixed precision
num_epochs = 100  # Adjust based on your needs
scaler = torch.cuda.amp.GradScaler()
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for (xi, xj) in dataloader:
        xi = xi.to(device, non_blocking=True)
        xj = xj.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            _, zi = model(xi)
            _, zj = model(xj)
            loss = criterion(zi, zj)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Save the encoder part of the model for feature extraction
torch.save(model.encoder.state_dict(), 'models/simclr_encoder.pth')
