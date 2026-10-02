import torch
import torch.nn as nn
from .sd_cro_resnet import SDCROResNet
from .encoders import RCAB


class FusionHead(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_blocks=4):
        super().__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.refine = nn.Sequential(
            *[RCAB(64) for _ in range(num_blocks)]
        )

        self.tail = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, f_res_3, f_ir_3):
        x = torch.cat([f_res_3, f_ir_3], dim=1)
        x = self.fusion_conv(x)
        x = self.refine(x)
        out = self.tail(x)
        return out


class SDCRONet_Stage2_LowDim(nn.Module):
    def __init__(self, stage1_checkpoint_path=None, encoder_channels=64):
        super().__init__()

        self.restoration_net = SDCROResNet(
            encoder_channels=encoder_channels,
            encoder_blocks=[2, 2, 4],
            decoder_blocks=[2, 2],
            num_atoms=16,
            degradation_dim=256
        )

        if stage1_checkpoint_path is not None:
            print(f"Loading Stage 1 weights from {stage1_checkpoint_path}...")
            checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu')
            if 'restoration_state_dict' in checkpoint:
                state_dict = checkpoint['restoration_state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.restoration_net.load_state_dict(new_state_dict, strict=True)
        else:
            print("No Stage 1 checkpoint provided. Skip Stage 1 init; expect full Stage2/Stage3 checkpoint to be loaded later.")

        self.compress_res = nn.Sequential(
            nn.Conv2d(encoder_channels, 3, 1),
            nn.Sigmoid()
        )
        self.compress_ir = nn.Sequential(
            nn.Conv2d(encoder_channels, 3, 1),
            nn.Sigmoid()
        )
        self.fusion_head = FusionHead(in_channels=3)

    def forward(self, infrared, degraded_visible):
        ir_feats = self.restoration_net.encoder_ir(infrared)
        vis_feats = self.restoration_net.encoder_vis(degraded_visible)

        ir_feat_shallow = ir_feats[0]
        ir_neck = ir_feats[-1]
        vis_neck = vis_feats[-1]

        _, struct_atoms = self.restoration_net.structure_domain(ir_neck)
        _, degradation_atoms = self.restoration_net.degradation_domain(degraded_visible)
        conflict_vector = self.restoration_net.conflict_map(vis_neck, struct_atoms, degradation_atoms)
        reasoning_vector = self.restoration_net.sd_cro(struct_atoms, degradation_atoms, conflict_vector)
        restored_neck = self.restoration_net.restoration_net(
            reasoning_vector, vis_neck, struct_atoms, degradation_atoms
        )

        img_restored, f_res = self.restoration_net.decoder(
            restored_neck, vis_feats[:-1], return_feat=True
        )

        f_res_3 = self.compress_res(f_res)
        f_ir_3 = self.compress_ir(ir_feat_shallow)

        img_fused = self.fusion_head(f_res_3, f_ir_3)

        return {
            'restored_image': img_restored,
            'fused_image': img_fused,
            'f_res_3': f_res_3,
            'f_ir_3': f_ir_3
        }