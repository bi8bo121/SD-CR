import torch
import torch.nn as nn
import torch.nn.functional as F

class ConflictMap_v2(nn.Module):

    def __init__(
        self,
        num_struct_atoms: int = 16,
        num_degrad_atoms: int = 16,
        channel_dim: int = 256
    ):
        super().__init__()
        self.num_struct_atoms = num_struct_atoms
        self.num_degrad_atoms = num_degrad_atoms
        self.channel_dim = channel_dim
        
        self.W_q = nn.Linear(channel_dim, channel_dim)
        
        self.P_s = nn.Parameter(torch.randn(num_struct_atoms, channel_dim) * 0.01)
        
        self.P_d = nn.Parameter(torch.randn(num_degrad_atoms, channel_dim) * 0.01)

        self.sigmoid = nn.Sigmoid()
        

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.P_s)
        nn.init.xavier_uniform_(self.P_d)
    
    def forward(
        self,
        visible_feat: torch.Tensor,
        struct_atoms: torch.Tensor,
        degradation_atoms: torch.Tensor
    ) -> torch.Tensor:
       
        b, c, h, w = visible_feat.shape
        k = struct_atoms.shape[1]
        m = degradation_atoms.shape[1]
    

        feat_spatial = visible_feat.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
        
        feat_proj = self.W_q(feat_spatial)  # [B, N, C]
        
        match_struct = torch.bmm(feat_proj, struct_atoms.permute(0, 2, 1))
        
        match_degrad = torch.bmm(feat_proj, degradation_atoms.permute(0, 2, 1))
        
        avg_match_struct = torch.mean(match_struct, dim=1)  # [B, K]
        avg_match_degrad = torch.mean(match_degrad, dim=1)  # [B, M]

        proj_s = torch.matmul(avg_match_struct, self.P_s)  # [B, C]

        proj_d = torch.matmul(avg_match_degrad, self.P_d)  # [B, C]

        conflict_raw = proj_s - proj_d  # [B, C]

        c = self.sigmoid(conflict_raw)  # [B, C]
        
        return c