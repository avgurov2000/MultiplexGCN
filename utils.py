import torch
from torch import nn

from typing import NamedTuple, Dict, List, Tuple
from collections import defaultdict


def explicit_broadcast(x, y):
    for _ in range(x.dim(), y.dim()):
        x = x.unsqueeze(-1)
    return x.expand_as(y) 

def from_edges_to_tensor(nx_network):
    edges_tensor = torch.tensor([e for e in nx_network.edges]).T
    edges_tensor = edges_tensor - edges_tensor.min()
    return edges_tensor