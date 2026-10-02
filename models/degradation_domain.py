import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DegradationSensorNetwork(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_conv_layers: int = 2,
        output_dim: int = 128
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        
        channels = in_channels
        for i in range(num_conv_layers):
            out_c = 32 * (i + 1)
            self.convs.append(
                nn.Conv2d(channels, out_c, kernel_size=3, padding=1, stride=1)
            )
            self.relus.append(nn.ReLU(inplace=True))
            channels = out_c

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        pool_out_dim = channels * 2 
        self.fc = nn.Sequential(
            nn.Linear(pool_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for conv, relu in zip(self.convs, self.relus):
            x = conv(x)
            x = relu(x)
        
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)

        pooled = torch.cat([avg_pool, max_pool], dim=1)
        

        d = self.fc(pooled)
        
        return d
    
class DegradationAtomGeneration(nn.Module):
    def __init__(
        self,
        degradation_vector_dim: int = 128,
        num_atoms: int = 16,
        channel_dim: int = 256
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.channel_dim = channel_dim
        
        self.W_d = nn.Linear(degradation_vector_dim, num_atoms * channel_dim)
        
        nn.init.xavier_uniform_(self.W_d.weight, gain=0.1)
        nn.init.zeros_(self.W_d.bias)
 
    def forward(self, d: torch.Tensor) -> torch.Tensor:
        b, l = d.shape
        d_norm = F.normalize(d, p=2, dim=1, eps=1e-8)
        d_mapped = self.W_d(d_norm * 0.1)

        D = d_mapped.view(b, self.num_atoms, self.channel_dim)
        return D

class DegradationDomainModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,          
        degradation_vector_dim: int = 128,
        channel_dim: int = 256,        
        num_atoms: int = 16,
        num_conv_layers: int = 2
    ):
        super().__init__()
        
        self.degradation_sensor = DegradationSensorNetwork(
            in_channels=in_channels,
            num_conv_layers=num_conv_layers,
            output_dim=degradation_vector_dim
        )

        self.atom_generator = DegradationAtomGeneration(
            degradation_vector_dim=degradation_vector_dim,
            num_atoms=num_atoms,
            channel_dim=channel_dim 
        )
    
    def forward(self, degraded_visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        d = self.degradation_sensor(degraded_visible) 
        D = self.atom_generator(d)
        return d, D
