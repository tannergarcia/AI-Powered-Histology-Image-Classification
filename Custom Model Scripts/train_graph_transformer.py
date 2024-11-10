import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load slide labels
metadata = pd.read_csv('data/BCC_labels.csv')  # Ensure this file contains 'slide_id' and 'label' columns

# Map labels to integers for binary classification
label_mapping = {
    'Clear': 0,
    'Present': 1
}
metadata['label'] = metadata['label'].map(label_mapping)

# Use 'StudyID #' as the grouping variable
# Ensure there are no missing values in 'StudyID #'
metadata = metadata.dropna(subset=['StudyID #'])
metadata['StudyID #'] = metadata['StudyID #'].astype(str)  # Convert to string if necessary

# Prepare data for splitting
X = metadata['slide_id']
y = metadata['label']
groups = metadata['StudyID #']  # Group by 'StudyID #'

# Split Data into Training and Test Sets
# Wel use StratifiedGroupKFold to ensure that the distribution of classes is maintained, and slides from the same study (patient) are not split between training and testing sets.
from sklearn.model_selection import StratifiedGroupKFold

n_splits = 5  # 5-fold cross-validation
skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

splits = list(skf.split(X, y, groups))

# For demonstration, we'll use the first fold for training and testing
train_idx, test_idx = splits[0]
train_slide_ids = X.iloc[train_idx].tolist()
test_slide_ids = X.iloc[test_idx].tolist()

# Create Data Loaders

from src.data_processing.graph_dataset import GraphDataset

# Create datasets
train_dataset = GraphDataset(root='data/graphs', slide_ids=train_slide_ids)
test_dataset = GraphDataset(root='data/graphs', slide_ids=test_slide_ids)

# Create data loaders
batch_size = 4  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Before the training loop in train_graph_transformer.py

# Analyze the sizes of graphs in the training dataset
num_nodes_list = []
num_edges_list = []

for data in train_dataset:
    num_nodes_list.append(data.num_nodes)
    num_edges_list.append(data.num_edges)

print(f"Number of graphs: {len(train_dataset)}")
print(f"Average number of nodes per graph: {sum(num_nodes_list)/len(num_nodes_list):.2f}")
print(f"Maximum number of nodes in a graph: {max(num_nodes_list)}")
print(f"Average number of edges per graph: {sum(num_edges_list)/len(num_edges_list):.2f}")
print(f"Maximum number of edges in a graph: {max(num_edges_list)}")


#  Set Up the Training Script

from src.models.graph_transformer_model import GraphTransformerModel
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
input_dim = 512  # Feature dimension from SimCLR encoder (adjust if different)
hidden_dim = 128  # As per the paper
num_classes = 2  # Binary classification: 'Present' or 'Clear'

model = GraphTransformerModel(input_dim, hidden_dim, num_classes)
model = model.to(device)

# Define optimizer and scheduler
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    scheduler.step()

    # Optionally, evaluate on the test set every few epochs
    if epoch % 10 == 0:
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                labels.extend(data.y.cpu().numpy())
        accuracy = accuracy_score(labels, preds)
        print(f'Validation Accuracy after Epoch {epoch}: {accuracy:.4f}')

# Save the final model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/graph_transformer_final.pth')