import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


from .msa import MSABlock

class LNI(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int,] = (32, 64, 128, 256),
        out_channels: Tuple[int,] = (32, 64, 128, 256),
        block_list: Tuple[int,] = (1, 1, 6, 2),
        dim_head: int = 32,
        ws_list: Tuple[int,] = (1, 2, 7, 7),
        proj_dropout: float = 0.0,
        drop_path_rates: Tuple[float] = (0.0,),
        mlp_ratio_list: Tuple[int,] = (4, 4, 4, 4),
        dropout: float = 0.0,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        
        self.fusion = nn.ModuleList([
            nn.Conv2d(inc, outc, kernel_size=1)
            for inc, outc in zip(in_channels, out_channels)
        ])
        
        # 空洞卷积层 
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(
                outc, 
                outc,
                kernel_size=3,
                padding=2**(i+1),
                dilation=2**(i+1),
                groups=outc
            ) for i, outc in enumerate(out_channels)
        ])
        
        # 特征交互模块
        self.interactions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(outc*2, outc, kernel_size=1),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True)
            ) for outc in out_channels[:-1]
        ])
        
        # MSA blocks for each scale
        self.blocks = nn.ModuleList([])
        for i, n_blocks in enumerate(block_list):
            blocks = []
            for j in range(n_blocks):
                blocks.append(
                    MSABlock(
                        in_dim=out_channels[i],
                        dim=out_channels[i],
                        heads=out_channels[i] // dim_head,
                        proj_dropout=proj_dropout,
                        mlp_ratio=mlp_ratio_list[i],
                        drop_path=drop_path_rates[j],
                        ws=ws_list[i],
                        with_cp=with_cp,
                    )
                )
            self.blocks.append(nn.ModuleList(blocks))
        
        self.norm = nn.ModuleList([
            nn.LayerNorm(outc) for outc in out_channels
        ])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        B = features[0].shape[0]
        
        # 1. 特征融合
        features = [
            fusion(x) for fusion, x in zip(self.fusion, features)
        ]
        
        # 2. 空洞卷积处理
        dilated_features = [
            conv(x) for conv, x in zip(self.dilated_convs, features)
        ]
   
        outputs = [features[0]]
        for i in range(len(features)-1):
           
            curr_feat = features[i]
            next_feat = features[i+1]
            
           
            fused = self.interactions[i](
                torch.cat([curr_feat, next_feat], dim=1)
            )
            outputs.append(fused)
            
   
        for i, (feature, blocks) in enumerate(zip(outputs, self.blocks)):
            H, W = feature.shape[-2:]
            x = feature.flatten(2).transpose(1, 2)
            
            # 应用MSA blocks
            for block in blocks:
                x = block(x, H, W)
                
        
            x = self.norm[i](x)
            
    
            outputs[i] = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
        return outputs