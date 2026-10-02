import torch
import torch.nn as nn
class SDCrossReasoningOperator_v2(nn.Module):
    def __init__(
        self,
        num_struct_atoms: int = 16,
        num_degrad_atoms: int = 16,
        channel_dim: int = 256 
    ):
        super().__init__()
        self.channel_dim = channel_dim

        self.W_s_score = nn.Parameter(torch.randn(channel_dim, 1) * 0.01)
        self.W_d_score = nn.Parameter(torch.randn(channel_dim, 1) * 0.01)

        self.gate_generator = nn.Sequential(
            nn.Linear(channel_dim, channel_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(channel_dim, channel_dim * 2), 
            nn.Sigmoid()
        )
        
        nn.init.xavier_uniform_(self.W_s_score)
        nn.init.xavier_uniform_(self.W_d_score)
    
    def forward(
        self,
        struct_atoms: torch.Tensor,    
        degradation_atoms: torch.Tensor, # [B, M, C]
        conflict_vector: torch.Tensor    
    ) -> torch.Tensor:
        
        struct_scores = torch.matmul(struct_atoms, self.W_s_score).squeeze(-1)
        w_s = torch.softmax(struct_scores, dim=1) 
        R_s = torch.bmm(w_s.unsqueeze(1), struct_atoms).squeeze(1) # [B, C]

        degrad_scores = torch.matmul(degradation_atoms, self.W_d_score).squeeze(-1)
        w_d = torch.softmax(degrad_scores, dim=1) 
        R_d = torch.bmm(w_d.unsqueeze(1), degradation_atoms).squeeze(1) # [B, C]

        gates = self.gate_generator(conflict_vector) # [B, 2C]
        gate_s, gate_d = torch.split(gates, self.channel_dim, dim=1) # [B, C], [B, C]
        
        R = gate_s * R_s + gate_d * R_d
        
        return R