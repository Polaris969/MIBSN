# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import logging
import math
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from torch import Tensor
from torch.types import _size
import torchvision

logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 10,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "msa_224": _cfg(),
}


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size**2, 
                                   kernel_size=kernel_size, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, 
                            kernel_size=kernel_size, padding=padding, groups=groups)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        return self.conv(torchvision.ops.deform_conv2d(
            x, offset, self.conv.weight, self.conv.bias, 
            padding=self.conv.padding[0]))


class MSAAttention(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 64,
        heads: int = 2,
        ws: int = 1,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
        with_cp: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0 and heads % 2 == 0, "dim should be divisible by heads and heads should be even"
        self.in_dim = in_dim
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.ws = ws
        self.with_cp = with_cp
        
        # IBPP路径参数
        p = int(2 ** (math.log2(dim//2)//2))
        self.p = p
        self.q = p
        self.ibpp_inner = nn.Linear(p, p)
        self.ibpp_outer = nn.Linear(p, p)
        self.ibpp_act = nn.SiLU()
        
        # 多维注意力参数
        self.to_qkv = nn.Linear(in_dim, 3 * dim)
        self.scale = qk_scale or self.dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.merge = nn.Linear(dim, dim)
        
        # 输出投影
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Dropout(proj_drop)
        )

    def _apply_ibpp(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # 结构化分解
        x = x.view(B, N, self.p, self.q)
        # 双重投影
        x = self.ibpp_act(self.ibpp_inner(x))
        x = self.ibpp_outer(x)
        return x.view(B, N, C)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B = x.shape[0]
        
        # 生成Q,K,V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 分为水平和垂直注意力头
        h_heads = v_heads = self.heads // 2
        q_h, q_v = q[:,:h_heads], q[:,h_heads:]
        k_h, k_v = k[:,:h_heads], k[:,h_heads:]
        v_h, v_v = v[:,:h_heads], v[:,h_heads:]
        
        # 水平注意力
        h_attn = (q_h @ k_h.transpose(-2, -1)) * self.scale
        h_attn = self.attend(h_attn)
        h_out = h_attn @ v_h
        
        # 垂直注意力  
        v_attn = (q_v @ k_v.transpose(-2, -1)) * self.scale
        v_attn = self.attend(v_attn)
        v_out = v_attn @ v_v
        
        # 合并水平和垂直注意力
        out = torch.cat([h_out, v_out], dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.merge(out)
        
        # IBPP路径
        ibpp_out = self._apply_ibpp(x)
        
        # 合并注意力和IBPP输出
        out = self.proj(torch.cat([out, ibpp_out], dim=-1))
        
        return out


class EHFF(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_cp = with_cp
        
        # 特征分割
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 两组可变形卷积
        self.deform_conv3 = DeformableConv2d(
            hidden_features//2, hidden_features//2, 
            kernel_size=3, padding=1
        )
        self.deform_conv5 = DeformableConv2d(
            hidden_features//2, hidden_features//2,
            kernel_size=5, padding=2
        )
        
        self.norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_func()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            x = self.fc1(x)
            B, N, C = x.shape
            
            # 分割特征
            x1, x2 = x.chunk(2, dim=-1)
            x1 = x1.transpose(1,2).view(B, C//2, H, W)
            x2 = x2.transpose(1,2).view(B, C//2, H, W)
            
            # 应用可变形卷积
            x1 = self.deform_conv3(x1)
            x2 = self.deform_conv5(x2)
            
            # 合并并处理
            x = torch.cat([x1, x2], dim=1)
            x = x.flatten(2).transpose(1,2)
            x = self.norm(x)
            x = self.act(x)
            x = self.fc2(x)
            
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class ScaleCouplingModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 使用1x1, 3x3, 5x5三个不同膨胀率的卷积
        self.conv1 = nn.Conv2d(dim, dim//3, kernel_size=1)
        self.conv3 = nn.Conv2d(dim, dim//3, kernel_size=3, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(dim, dim//3, kernel_size=5, padding=2, dilation=1)
        
        self.fusion = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        
        x = torch.cat([x1, x3, x5], dim=1)
        x = self.fusion(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.act(x)
        return x


class MSABlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 64,
        heads: int = 2,
        proj_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        ws: int = 1,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp

        # 多维自注意力
        self.attn_norm = nn.LayerNorm(in_dim)
        self.attn = MSAAttention(
            in_dim=in_dim,
            dim=dim,
            heads=heads,
            ws=ws,
            proj_drop=proj_dropout,
            with_cp=with_cp,
        )
        self.attn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 混合增强前馈网络
        self.ffn_norm = nn.LayerNorm(in_dim)
        self.ffn = EHFF(
            in_features=in_dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_func=nn.GELU,
            with_cp=with_cp,
        )
        self.ffn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        # 注意力模块
        res = x
        x = self.attn_norm(x)
        x = self.attn(x, H, W)
        x = self.attn_drop_path(x).add(res)

        # 前馈网络模块
        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x, H, W)
        x = self.ffn_drop_path(x).add(res)

        return x


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
        
        # 特征融合层 - 使用1x1卷积融合特征
        self.fusion = nn.ModuleList([
            nn.Conv2d(inc, outc, kernel_size=1)
            for inc, outc in zip(in_channels, out_channels)
        ])
        
        # 空洞卷积层 - 用于特征交互
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
        
        # 3. 级联特征交互
        outputs = [features[0]]
        for i in range(len(features)-1):
            # 当前尺度特征与下一尺度特征交互
            curr_feat = features[i]
            next_feat = features[i+1]
            
            # 特征融合
            fused = self.interactions[i](
                torch.cat([curr_feat, next_feat], dim=1)
            )
            outputs.append(fused)
            
        # 4. MSA处理
        for i, (feature, blocks) in enumerate(zip(outputs, self.blocks)):
            H, W = feature.shape[-2:]
            x = feature.flatten(2).transpose(1, 2)
            
            # 应用MSA blocks
            for block in blocks:
                x = block(x, H, W)
                
            # 应用norm
            x = self.norm[i](x)
            
            # 重塑回空间维度
            outputs[i] = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
        return outputs


class MSA(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        stride: int = 4,
        channels: int = 64,
        channel_list: Tuple[Tuple[int,],] = (
            (32,),
            (32, 64),
            (32, 64, 128),
            (32, 64, 128),
            (32, 64, 128, 256),
            (32, 64, 128, 256),
        ),
        block_list: Tuple[Tuple[int]] = (
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 2),
            (1, 1, 2, 2),
        ),
        dim_head: int = 32,
        ws_list: Tuple[int,] = (1, 2, 7, 7),
        proj_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        mlp_ratio_list: Tuple[int,] = (4, 4, 4, 4),
        dropout: float = 0.0,
        num_classes: int = 10,
        head_dropout: float = 0.1,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        
        # 计算drop path rates
        total_blocks = sum(max(b) for b in block_list)
        total_drop_path_rates = (
            torch.linspace(0, drop_path_rate, total_blocks).numpy().tolist()
        )
        
        # 构建主干网络
        self.features = []
        cur = 0
        self.channel_list = channel_list = [[channels]] + list(channel_list)
        
        # 构建stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 构建主干
        for i, blocks in enumerate(block_list):
            inc, outc = channel_list[i:i+2]
            depth_per_stage = max(blocks)
            
            # 添加尺度耦合模块
            self.features.append(
                ScaleCouplingModule(dim=inc[-1])
            )
            
            # 添加邻域特征交互模块
            self.features.append(
                LNI(
                    in_channels=inc,
                    out_channels=outc,
                    block_list=blocks,
                    dim_head=dim_head,
                    ws_list=ws_list,
                    proj_dropout=proj_dropout,
                    drop_path_rates=total_drop_path_rates[cur:cur+depth_per_stage],
                    mlp_ratio_list=mlp_ratio_list,
                    dropout=dropout,
                    with_cp=with_cp,
                )
            )
            cur += depth_per_stage
            
        self.features = nn.Sequential(*self.features)
        
        # 构建分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_list[-1][-1], num_classes)
        )
        
        self.head_dropout = head_dropout
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
            
    def forward_features(self, x: Tensor) -> List[Tensor]:
        # stem处理
        x = self.stem(x)
        # 主干网络处理
        x = self.features([x])
        return x
        
    def forward(self, x: Tensor) -> Tensor:
        # 特征提取
        x = self.forward_features(x)[-1]  # 取最后一个尺度的特征
        # 全局池化
        x = self.avgpool(x)
        # 分类头
        x = self.head(x)
        return x


