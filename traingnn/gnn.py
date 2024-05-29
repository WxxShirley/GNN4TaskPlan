import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, GINConv
import torch.nn as nn
import torch


def build_conv(conv_type: str):
    """Return the specific gnn as`conv_type`"""
    if conv_type == "GCN":
        return GCNConv
    elif conv_type == "GIN":
        return lambda i, h: GINConv(
            nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))
        )
    elif conv_type == "GAT":
        return GATConv
    elif conv_type == "TransformerConv":
        return TransformerConv
    elif conv_type == "SAGE":
        return SAGEConv
    else:
        raise KeyError("GNN_TYPE can only be GAT, GCN, SAGE, GIN, and TransformerConv")


class SGC(nn.Module):
    def __init__(self):
        super().__init__()

        init_alpha = torch.Tensor([1.0])
        self.alpha = nn.Parameter(init_alpha)

    def forward(self, x, sp_edge_index):
        neighbor_emb = torch.sparse.mm(sp_edge_index, x)
        
        # TODO: check the value of \alpha in [0, 1]
        final_emb = self.alpha * x + (1.0 - self.alpha) * neighbor_emb
        return final_emb


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, gnn_type="GAT", dropout=0.0):
        super().__init__()

        conv = build_conv(gnn_type)

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.LeakyReLU()
        
        if n_layers == 1:
            self.conv_layers = nn.ModuleList([conv(input_dim, hidden_dim)])
        elif n_layers == 2:
            self.conv_layers = nn.ModuleList([conv(input_dim, hidden_dim), conv(hidden_dim, hidden_dim)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers-1)])
        
    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, x, edge_index):
        layer_node_emb = [x]
        for i, graph_conv in enumerate(self.conv_layers[:-1]):
            x = graph_conv(x, edge_index)
            x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        node_emb = self.conv_layers[-1](x, edge_index)
        layer_node_emb.append(node_emb)
        
        return torch.mean(torch.stack(layer_node_emb), dim=0)
