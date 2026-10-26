"""Jiang et al. GLCN (CVPR 2019), adapted from node classification to retrieval.

Equations documented in https://arxiv.org/pdf/1811.09971:
projected absolute differences + ReLU + masked row-softmax (9--10),
graph smoothness/Frobenius regularization (11), propagation S X W (13).
Point graphs and mean pooling are sketch-specific adaptations.
"""
import torch
from torch import nn
import torch.nn.functional as F

from encoders.gat import build_sketch_graph


class GraphLearningLayer(nn.Module):
    def __init__(self, input_dim=4, graph_dim=2, gamma=1.):
        super().__init__()
        self.projection = nn.Linear(input_dim, graph_dim, bias=False)
        self.metric = nn.Parameter(torch.empty(graph_dim))
        self.gamma = gamma
        nn.init.xavier_uniform_(self.projection.weight)
        # Avoid initializing every pair in the dead half of ReLU.
        nn.init.uniform_(self.metric, 0., 1. / graph_dim)

    def forward(self, features, adjacency, valid, regularize=True):
        z = self.projection(features.masked_fill(~valid.unsqueeze(-1), 0.))
        differences = z.unsqueeze(2) - z.unsqueeze(1)
        logits = F.relu((differences.abs() * self.metric).sum(-1))
        real_edges = valid.unsqueeze(1) & valid.unsqueeze(2)
        allowed = adjacency & real_edges
        # Dummy padding self-loops avoid all-masked softmax; remove them afterwards.
        eye = torch.eye(features.shape[1], dtype=torch.bool, device=features.device)[None]
        allowed = allowed | eye
        graph = logits.masked_fill(~allowed, float('-inf')).softmax(dim=-1)
        graph = graph.masked_fill(~real_edges, 0.)
        penalty = features.new_zeros(())
        if regularize:
            smoothness = (differences.square().sum(-1) * graph).sum(dim=(1, 2))
            frobenius = graph.square().sum(dim=(1, 2))
            # Normalize per real node for variable-length sketch graphs, then batch mean.
            penalty = ((smoothness + self.gamma * frobenius) / valid.sum(1).clamp_min(1)).mean()
        return graph, penalty


class GLCNSketchEncoder(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=64, graph_dim=2, neighbors=8,
                 dropout=0.1, graph_loss_weight=0.01, gamma=1.):
        super().__init__()
        if min(embed_dim, hidden_dim, graph_dim) < 1 or neighbors < 0 or min(graph_loss_weight, gamma) < 0:
            raise ValueError('Invalid GLCN dimensions or regularization parameters')
        self.neighbors = neighbors
        self.graph_loss_weight = graph_loss_weight
        self.graph_learning = GraphLearningLayer(4, graph_dim, gamma)
        self.conv1 = nn.Linear(4, hidden_dim, bias=False)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(hidden_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        self._auxiliary_loss = None

    def forward(self, sketch):
        features, adjacency, valid = build_sketch_graph(sketch, self.neighbors)
        regularize = self.training and torch.is_grad_enabled()
        graph, penalty = self.graph_learning(features, adjacency, valid, regularize)
        self._auxiliary_loss = self.graph_loss_weight * penalty if regularize else None
        # S is row-stochastic on real nodes, so D_s = I in the paper's normalization.
        hidden = F.relu(graph @ self.conv1(self.dropout(features)))
        hidden = F.relu(graph @ self.conv2(self.dropout(hidden)))
        hidden = hidden.masked_fill(~valid.unsqueeze(-1), 0.)
        pooled = hidden.sum(1) / valid.sum(1, keepdim=True)
        return F.normalize(self.readout(pooled), dim=-1)

    def auxiliary_loss(self):
        return self._auxiliary_loss
