
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class JointRestorationLoss_v2(nn.Module):
    
    def __init__(
        self,
        weight_l1: float = 1.0,                   

        weight_struct_reg: float = 0.1,    
        weight_degrad_reg: float = 0.1
    ):
        super().__init__()
        self.weight_l1 = weight_l1

        self.weight_struct_reg = weight_struct_reg
        self.weight_degrad_reg = weight_degrad_reg
        
        self.l1_loss = nn.L1Loss()
    
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        struct_atoms: torch.Tensor,
        degradation_atoms: torch.Tensor,
        alpha_s: torch.Tensor = None,
        alpha_d: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        loss_l1 = self.weight_l1 * self.l1_loss(pred, target)
        
        struct_norms = torch.norm(struct_atoms, dim=2)
        loss_struct_reg = self.weight_struct_reg * torch.mean((struct_norms - 1.0) ** 2)
        
        degrad_norms = torch.norm(degradation_atoms, dim=2)
        loss_degrad_reg = self.weight_degrad_reg * torch.mean((degrad_norms - 1.0) ** 2)

        total_loss = (
            loss_l1 + 
            loss_struct_reg + 
            loss_degrad_reg
        )

        return total_loss, {
            'loss': float(total_loss.item()),
            'l1_loss': float(loss_l1.item()),
            'loss_struct_reg': float(loss_struct_reg.item()),
            'loss_degrad_reg': float(loss_degrad_reg.item())
        }
