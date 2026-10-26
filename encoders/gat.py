"""Original GAT attention (Velickovic et al., ICLR 2018), adapted to sketch retrieval.

Reference: https://arxiv.org/abs/1710.10903, equations (3)--(6).
Point graph construction and graph-level pooling are adaptations, not paper benchmarks.
Pure PyTorch dense masked attention; intended for bounded (352-point) sketches.
"""
import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Multi-head a^T[Wh_i || Wh_j] attention over an explicit adjacency mask."""

    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.projection = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_source = nn.Parameter(torch.empty(heads, out_features))
        self.attn_neighbor = nn.Parameter(torch.empty(heads, out_features))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_neighbor)

    def forward(self, features, adjacency, valid):
        batch, nodes, _ = features.shape
        h = self.projection(F.dropout(features, self.dropout, self.training))
        h = h.view(batch, nodes, self.heads, self.out_features).transpose(1, 2)
        source = (h * self.attn_source[None, :, None]).sum(-1)
        neighbor = (h * self.attn_neighbor[None, :, None]).sum(-1)
        logits = F.leaky_relu(source.unsqueeze(-1) + neighbor.unsqueeze(-2), negative_slope=0.2)
        allowed = adjacency & valid.unsqueeze(1) & valid.unsqueeze(2)
        # Self loops keep even isolated nodes well defined. Padding rows are zeroed below.
        allowed = allowed | torch.eye(nodes, dtype=torch.bool, device=h.device).unsqueeze(0)
        attention = logits.masked_fill(~allowed.unsqueeze(1), float('-inf')).softmax(dim=-1)
        attention = F.dropout(attention, self.dropout, self.training)
        output = attention @ h
        if self.concat:
            output = output.transpose(1, 2).reshape(batch, nodes, -1)
        else:
            output = output.mean(dim=1)
        return output.masked_fill(~valid.unsqueeze(-1), 0.)


class GATSketchEncoder(nn.Module):
    """S5 [B,N,5] -> graph -> normalized retrieval embedding [B,embed_dim].

    Uses the project's REL S5 convention: channels 2/3 are pen-up/down;
    their sum identifies real points (including a real point at the final slot).
    The existing EOS channel alone does not identify padding in this project.
    """

    def __init__(self, embed_dim=512, hidden_dim=32, heads=4, neighbors=8, dropout=0.1):
        super().__init__()
        if neighbors < 0 or heads < 1 or hidden_dim < 1 or embed_dim < 1:
            raise ValueError('Invalid GAT dimensions or neighbor count')
        self.neighbors = neighbors
        self.gat1 = GraphAttentionLayer(4, hidden_dim, heads, True, dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim * heads, hidden_dim * 2, heads, False, dropout)
        self.readout = nn.Sequential(nn.Linear(hidden_dim * 2, embed_dim), nn.LayerNorm(embed_dim))

    def build_graph(self, sketch):
        return build_sketch_graph(sketch, self.neighbors)

    def forward(self, sketch):
        features, adjacency, valid = self.build_graph(sketch)
        hidden = F.elu(self.gat1(features, adjacency, valid))
        hidden = self.gat2(hidden, adjacency, valid)
        pooled = hidden.sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return F.normalize(self.readout(pooled), p=2, dim=-1)


def build_sketch_graph(sketch, neighbors=8):
    if sketch.ndim != 3 or sketch.shape[-1] != 5 or sketch.shape[1] == 0:
        raise ValueError('Graph encoder expects non-empty REL S5 input [B,N,5]')
    valid = sketch[..., 2:4].sum(-1) > 0
    if not valid.any(dim=1).all():
        raise ValueError('Graph encoder cannot encode a sketch with no valid points')
    xy = sketch[..., :2].masked_fill(~valid.unsqueeze(-1), 0.).cumsum(dim=1)
    count = valid.sum(dim=1, keepdim=True).unsqueeze(-1)
    center = (xy * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / count
    xy = (xy - center).masked_fill(~valid.unsqueeze(-1), 0.)
    scale = xy.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    xy = xy / scale
    features = torch.cat((xy, sketch[..., 2:4]), dim=-1).masked_fill(~valid.unsqueeze(-1), 0.)
    batch, nodes, _ = sketch.shape
    eye = torch.eye(nodes, dtype=torch.bool, device=sketch.device).unsqueeze(0)
    real_edges = valid.unsqueeze(1) & valid.unsqueeze(2)
    adjacency = eye.expand(batch, -1, -1).clone()
    if neighbors and nodes > 1:
        distance = torch.cdist(xy, xy).masked_fill(~real_edges | eye, float('inf'))
        indices = distance.topk(min(neighbors, nodes - 1), largest=False).indices
        spatial = torch.zeros_like(adjacency).scatter_(2, indices, True)
        adjacency |= (spatial | spatial.transpose(1, 2)) & real_edges
    # Consecutive points connect only when the preceding point has pen down.
    if nodes > 1:
        indices = torch.arange(nodes - 1, device=sketch.device)
        consecutive = valid[:, :-1] & valid[:, 1:] & (sketch[:, :-1, 3] > 0)
        adjacency[:, indices, indices + 1] |= consecutive
        adjacency[:, indices + 1, indices] |= consecutive
    return features, adjacency & real_edges, valid

