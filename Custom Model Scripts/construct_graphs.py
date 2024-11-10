# construct_graphs.py

import pandas as pd
import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

# Load features
df = pd.read_csv('data/features/patch_features.csv')

# Convert feature columns to numeric types
feature_cols = [str(i) for i in range(512)]
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

# Handle missing values
if df[feature_cols].isnull().values.any():
    print("Warning: NaNs detected in feature columns. Filling NaNs with zero.")
    df[feature_cols] = df[feature_cols].fillna(0)

# Load slide labels
metadata = pd.read_csv('data/BCC_labels.csv')  # Columns: slide_id, label
labels_dict = dict(zip(metadata['slide_id'], metadata['label']))  # 'Clear' or 'Present'

# Map labels to integers
label_mapping = {'Clear': 0, 'Present': 1}
df['label'] = df['slide_id'].map(labels_dict).map(label_mapping)

# Create directory to save individual graphs
os.makedirs('data/graphs', exist_ok=True)

# For each slide, create a graph where nodes are patches and edges connect neighboring patches.
for slide_id, group in df.groupby('slide_id'):
    # Skip slides without labels
    if slide_id not in labels_dict:
        continue

    # Extract features and coordinates for this slide
    features_array = group[feature_cols].values.astype(np.float32)
    coords_array = group[['x_coord', 'y_coord']].values.astype(int)

    # Create a graph
    G = nx.Graph()

    # Add nodes
    node_ids = group.index.tolist()
    for i, node_id in enumerate(node_ids):
        G.add_node(node_id, feature=features_array[i], coords=coords_array[i])

    # Use coordinates directly as grid positions
    grid_positions = [tuple(pos) for pos in coords_array]

    # Create mapping from grid positions to node indices
    grid_to_node = {pos: idx for idx, pos in enumerate(grid_positions)}

    # Add edges to immediate neighbors
    for i, node_id1 in enumerate(node_ids):
        pos = grid_positions[i]
        neighbors = [
            (pos[0] - 1, pos[1]),     # Left
            (pos[0] + 1, pos[1]),     # Right
            (pos[0], pos[1] - 1),     # Above
            (pos[0], pos[1] + 1),     # Below
            (pos[0] - 1, pos[1] - 1), # Top-left
            (pos[0] - 1, pos[1] + 1), # Bottom-left
            (pos[0] + 1, pos[1] - 1), # Top-right
            (pos[0] + 1, pos[1] + 1)  # Bottom-right
        ]
        for neighbor_pos in neighbors:
            if neighbor_pos in grid_to_node:
                neighbor_idx = grid_to_node[neighbor_pos]
                node_id2 = node_ids[neighbor_idx]
                G.add_edge(node_id1, node_id2)

    # Output the number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    print(f"Graph for slide_id: {slide_id}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

    # Convert to PyTorch Geometric Data object
    data = from_networkx(G)

    # Assign features tensor
    data.x = torch.tensor(features_array, dtype=torch.float32)
    
    # Convert edge_index to long tensor
    data.edge_index = data.edge_index.long()

    # Assign labels
    data.y = torch.tensor([label_mapping[labels_dict[slide_id]]], dtype=torch.long)
    data.slide_id = slide_id
    data.num_nodes = data.x.shape[0]

    # Save the graph Data object to a file
    graph_file = f'data/graphs/{slide_id}.pt'
    torch.save(data, graph_file)
