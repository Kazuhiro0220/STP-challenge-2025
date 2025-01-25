import torch
import torch.nn as nn

from rnn.utils import get_params_str, cudafy_list
from rnn.utils import nll_gauss
from rnn.utils import batch_error, roll_out, roll_out_test
import torch.nn.functional as F


# gnn.py または rnn.py に追加

class InteractionNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.gat1 = GATConv(node_dim, 64, heads=4)
        self.gat2 = GATConv(64*4, 32)
        self.fc = nn.Linear(32, node_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        return self.fc(x)