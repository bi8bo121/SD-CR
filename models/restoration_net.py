import torch
import torch.nn as nn
import torch.nn.functional as F

class RestorationIntensityMapping(nn.Module):

    
    def __init__(self, channel_dim: int = 256):
     
        super().__init__()
        self.channel_dim = channel_dim

        self.w_s = nn.Parameter(torch.randn(channel_dim) * 0.01)
        self.w_d = nn.Parameter(torch.randn(channel_dim) * 0.01)

        self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_uniform_(self.w_s.unsqueeze(0))
        nn.init.xavier_uniform_(self.w_d.unsqueeze(0))
    
    def forward(self, R: torch.Tensor) -> tuple:

        alpha_s_logits = torch.matmul(R, self.w_s)  # [B]
        alpha_s = self.sigmoid(alpha_s_logits)  # [B]

        alpha_d_logits = torch.matmul(R, self.w_d)  # [B]
        alpha_d = self.sigmoid(alpha_d_logits)  # [B]
        
        return alpha_s, alpha_d


class StructureCompensationBranch(nn.Module):
  
    
    def __init__(self, channel_dim: int =256):
      
        super().__init__()
        self.channel_dim = channel_dim

        self.W_s2f = nn.Linear(channel_dim, channel_dim)
        
        self.W_sq = nn.Linear(channel_dim, channel_dim)
        self.W_sk = nn.Linear(channel_dim, channel_dim)
        self.W_sv = nn.Linear(channel_dim, channel_dim)
        
        for module in [self.W_s2f, self.W_sq, self.W_sk, self.W_sv]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        visible_feat: torch.Tensor,
        struct_atoms: torch.Tensor
    ) -> torch.Tensor:
       
        b, c, h, w = visible_feat.shape
        k = struct_atoms.shape[1]  # K
        
        feat_spatial = visible_feat.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
        n = h * w

        S_hat = self.W_s2f(struct_atoms)  # [B, K, C]
        
        # Q = F_v W_q: [B, N, C] x [C, C] -> [B, N, C]
        Q = self.W_sq(feat_spatial)  # [B, N, C]
        
        # K = Ŝ W_sk: [B, K, C] x [C, C] -> [B, K, C]
        K = self.W_sk(S_hat)  # [B, K, C]

        # [B, N, C] x [B, C, K] -> [B, N, K]
        scores = torch.bmm(Q, K.permute(0, 2, 1)) / (c ** 0.5)  # [B, N, K]
        attention = torch.softmax(scores, dim=2)  # [B, N, K]
        
        # V = Ŝ W_sv: [B, K, C] x [C, C] -> [B, K, C]
        V = self.W_sv(S_hat)  # [B, K, C]

        # [B, N, K] x [B, K, C] -> [B, N, C]
        F_struct_spatial = torch.bmm(attention, V)  # [B, N, C]

        F_struct = F_struct_spatial.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        
        return F_struct


class DegradationRepairBranch(nn.Module):
    
    def __init__(self, channel_dim: int = 256):
       
        super().__init__()
        self.channel_dim = channel_dim
        
        self.W_d2f = nn.Linear(channel_dim, channel_dim)

        self.W_dq = nn.Linear(channel_dim, channel_dim)
        self.W_dk = nn.Linear(channel_dim, channel_dim)
        self.W_dv = nn.Linear(channel_dim, channel_dim)

        for module in [self.W_d2f, self.W_dq, self.W_dk, self.W_dv]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        visible_feat: torch.Tensor,
        degradation_atoms: torch.Tensor
    ) -> torch.Tensor:
  
        b, c, h, w = visible_feat.shape
        m = degradation_atoms.shape[1]  # M

        feat_spatial = visible_feat.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
        n = h * w

        # D̂ = D W_d2f: [B, M, C] x [C, C] -> [B, M, C]
        D_hat = self.W_d2f(degradation_atoms)  # [B, M, C]

        # Q = F_v W_dq: [B, N, C] x [C, C] -> [B, N, C]
        Q = self.W_dq(feat_spatial)  # [B, N, C]
        
        # K = D̂ W_dk: [B, M, C] x [C, C] -> [B, M, C]
        K = self.W_dk(D_hat)  # [B, M, C]
        
        # [B, N, C] x [B, C, M] -> [B, N, M]
        scores = torch.bmm(Q, K.permute(0, 2, 1)) / (c ** 0.5)  # [B, N, M]
        attention = torch.softmax(scores, dim=2)  # [B, N, M]
        
        # V = D̂ W_dv: [B, M, C] x [C, C] -> [B, M, C]
        V = self.W_dv(D_hat)  # [B, M, C]

        # [B, N, M] x [B, M, C] -> [B, N, C]
        F_deg_spatial = torch.bmm(attention, V)  # [B, N, C]

        F_deg = F_deg_spatial.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
        
        return F_deg


class RestorationGuidedRestorationNetwork(nn.Module):
 
    
    def __init__(self, channel_dim: int =256 ):
      
        super().__init__()
        self.channel_dim = channel_dim
        
        self.intensity_mapping = RestorationIntensityMapping(channel_dim)

        self.struct_branch = StructureCompensationBranch(channel_dim)

        self.degrad_branch = DegradationRepairBranch(channel_dim)
    
    def forward(
        self,
        R: torch.Tensor,
        visible_feat: torch.Tensor,
        struct_atoms: torch.Tensor,
        degradation_atoms: torch.Tensor
    ) -> torch.Tensor:
    
        alpha_s, alpha_d = self.intensity_mapping(R)  # [B], [B]
        F_struct = self.struct_branch(visible_feat, struct_atoms)  # [B, C, H, W]
        F_deg = self.degrad_branch(visible_feat, degradation_atoms)  # [B, C, H, W]
        
        alpha_s = alpha_s.view(-1, 1, 1, 1)
        alpha_d = alpha_d.view(-1, 1, 1, 1)
        
        F_res = visible_feat + alpha_s * F_struct + alpha_d * F_deg
        
        return F_res