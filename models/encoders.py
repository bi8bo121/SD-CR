import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import torch
import torch.nn as nn

def norm_layer(channels):
    return nn.GroupNorm(8, channels)
class RCAB(nn.Module):
    def __init__(self, dim, reduction=8):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.PReLU(), 
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.body(x)
        res = res * self.ca(res)
        return x + res 


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, num_layers=[2, 2, 4]):

        super(Encoder, self).__init__()
        
        self.head = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        
        self.stage1 = self._make_layer(base_dim, num_layers[0])
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1)

        self.stage2 = self._make_layer(base_dim * 2, num_layers[1])
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1) 

        self.stage3 = self._make_layer(base_dim * 4, num_layers[2])

    def _make_layer(self, dim, blocks):
        layers = [RCAB(dim) for _ in range(blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)

        feat1 = self.stage1(x)     
        x = self.down1(feat1)

        feat2 = self.stage2(x)      
        x = self.down2(feat2)

        feat_out = self.stage3(x)  
        
        return [feat1, feat2, feat_out]


class Decoder(nn.Module):
    def __init__(self, out_channels=3, base_dim=64, num_layers=[2, 2]):

        super(Decoder, self).__init__()
        

        self.up1 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 8, 3, 1, 1), 
            nn.PixelShuffle(2)
        )
        self.stage1 = self._make_layer(base_dim * 2, num_layers[0])

        self.up2 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.stage2 = self._make_layer(base_dim, num_layers[1])

        self.tail = nn.Conv2d(base_dim, out_channels, 3, 1, 1)

    def _make_layer(self, dim, blocks):
        layers = [RCAB(dim) for _ in range(blocks)]
        return nn.Sequential(*layers)
   
    def forward(self, x, skips, return_feat=False): 

        skip_shallow = skips[0]
        skip_middle = skips[1]

        x = self.up1(x)             
        x = x + skip_middle         
        x = self.stage1(x)

        x = self.up2(x)             
        x = x + skip_shallow        
        x = self.stage2(x) 
        
        feat_res = x 

        out = self.tail(x)          
        
        if return_feat:
            return out, feat_res 
        else:
            return out
