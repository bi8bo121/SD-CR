import torch
import torch.nn as nn
from typing import Dict, List

from .encoders import Encoder, Decoder
from .structure_domain_v2 import StructureDomainModule
from .degradation_domain import DegradationDomainModule
from .conflict_map_v2 import ConflictMap_v2
from .sd_cro_v2 import SDCrossReasoningOperator_v2
from .restoration_net import RestorationGuidedRestorationNetwork

class SDCROResNet(nn.Module):
    
    def __init__(
        self,
        encoder_channels: int = 64, 
        encoder_blocks: list = [2, 2, 4], 
        decoder_blocks: list = [2, 2],   
        num_atoms: int = 16,
        degradation_dim: int = 256,
        **kwargs
    ):
        super().__init__()
        
        self.bottleneck_dim = encoder_channels * 4 

        self.encoder_ir = Encoder(
            in_channels=1,
            base_dim=encoder_channels, 
            num_layers=encoder_blocks 
        )
        
        self.encoder_vis = Encoder(
            in_channels=3,
            base_dim=encoder_channels,
            num_layers=encoder_blocks
        )
        

        self.structure_domain = StructureDomainModule(
            channel_dim=self.bottleneck_dim,
            num_atoms=num_atoms
        )
        

        self.degradation_domain = DegradationDomainModule(
            in_channels=3,
            degradation_vector_dim=degradation_dim,
            channel_dim=self.bottleneck_dim,
            num_atoms=num_atoms,
            num_conv_layers=2
        )
  
        self.conflict_map = ConflictMap_v2(
            num_struct_atoms=num_atoms,
            num_degrad_atoms=num_atoms,
            channel_dim=self.bottleneck_dim 
        )
        
        self.sd_cro = SDCrossReasoningOperator_v2(
            num_struct_atoms=num_atoms,
            num_degrad_atoms=num_atoms,
            channel_dim=self.bottleneck_dim 
        )
        

        self.restoration_net = RestorationGuidedRestorationNetwork(
            channel_dim=self.bottleneck_dim 
        )
        

        self.decoder = Decoder(
            out_channels=3,
            base_dim=encoder_channels, 
            num_layers=decoder_blocks
        )
    
    def forward(
        self,
        infrared: torch.Tensor,
        degraded_visible: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        ir_feats = self.encoder_ir(infrared)
        vis_feats = self.encoder_vis(degraded_visible)
        
        ir_neck = ir_feats[-1]    
        vis_neck = vis_feats[-1] 
        
        skip_connections = vis_feats[:-1] 

        struct_feat, struct_atoms = self.structure_domain(ir_neck)

        degradation_vector, degradation_atoms = self.degradation_domain(degraded_visible)
 
        conflict_vector = self.conflict_map(vis_neck, struct_atoms, degradation_atoms)
 
        reasoning_vector = self.sd_cro(
            struct_atoms,
            degradation_atoms,
            conflict_vector
        )
        

        restored_neck = self.restoration_net(
            reasoning_vector,
            vis_neck,
            struct_atoms,
            degradation_atoms
        )
        

        restored_image = self.decoder(restored_neck, skip_connections)

        final_image = restored_image + degraded_visible

        return {
            'restored_image': final_image,     

            'ir_feat': ir_neck,
            'vis_feat': vis_neck,
            'struct_feat': struct_feat,
            'restored_feat': restored_neck,
            
            'struct_atoms': struct_atoms,
            'degradation_atoms': degradation_atoms,
            
            'degradation_vector': degradation_vector,
            'conflict_vector': conflict_vector,
            'reasoning_vector': reasoning_vector
        }