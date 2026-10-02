import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class StructureExtractor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.extract = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid() 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        struct_attn = self.extract(x)
        return x * struct_attn

class StructureAtomAggregation(nn.Module):
    def __init__(self, num_atoms: int, channel_dim: int):
        super().__init__()
        self.num_atoms = num_atoms
        self.channel_dim = channel_dim
        
        self.structure_queries = nn.Parameter(
            torch.randn(num_atoms, channel_dim) * 0.01
        )
        nn.init.xavier_uniform_(self.structure_queries)
    
    def forward(self, struct_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = struct_feat.shape
        
        feat_spatial = struct_feat.view(b, c, -1).permute(0, 2, 1)
        similarity = torch.matmul(feat_spatial, self.structure_queries.t())
        weight_ak = torch.softmax(similarity / np.sqrt(c), dim=1)
        struct_atoms = torch.bmm(weight_ak.permute(0, 2, 1), feat_spatial)
        
        return struct_atoms

class StructureDomainModule(nn.Module):
    def __init__(
        self,
        channel_dim: int = 256, 
        num_atoms: int = 16
    ):
        super().__init__()
        self.channel_dim = channel_dim
        
        self.struct_enhancer = StructureExtractor(channel_dim)
        self.struct_aggregator = StructureAtomAggregation(num_atoms, channel_dim)
    
    def forward(self, ir_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        struct_feat = self.struct_enhancer(ir_feat)
        struct_atoms = self.struct_aggregator(struct_feat)
        
        return struct_feat, struct_atoms