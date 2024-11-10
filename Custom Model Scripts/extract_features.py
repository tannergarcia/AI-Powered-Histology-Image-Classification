import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from src.models.simclr_model import SimCLRModel
from src.data_processing.feature_extraction_dataset import FeatureExtractionDataset
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained encoder
encoder = SimCLRModel(base_model='resnet18').encoder
encoder.load_state_dict(torch.load('models/simclr_encoder.pth'))
encoder = encoder.to(device)
encoder.eval()

# Define transform without augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize dataset and dataloader
dataset = FeatureExtractionDataset(root_dir='data/patches/BCC', transform=transform)
batch_size = 256  # Adjust based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Extract features
features = []
patch_info = []
with torch.no_grad():
    for images, img_paths in dataloader:
        images = images.to(device, non_blocking=True)
        h = encoder(images)
        h = h.cpu()
        features.append(h)
        patch_info.extend(img_paths)

# Concatenate features
features = torch.cat(features, dim=0).numpy()

# Create a DataFrame to store features and patch information
df = pd.DataFrame(features)
df['patch_path'] = patch_info

# Extract slide IDs and coordinates from patch paths
df['slide_id'] = df['patch_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
df['x_coord'] = df['patch_path'].apply(lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
df['y_coord'] = df['patch_path'].apply(lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))

# Save the features to a CSV file
df.to_csv('data/features/patch_features.csv', index=False)
