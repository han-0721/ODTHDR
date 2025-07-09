import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from typing import Tuple
from math import prod
import os
import sys
sys.path.append(os.getcwd())
import yaml
import copy

class ResBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, res_scale=1.0, bias=True, shortcut=True):
        super().__init__()
        self.res_scale = res_scale
        self.shortcut = shortcut
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return out * self.res_scale

def window_partition(x, window_size: Tuple[int, int]):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C))
    return windows

def window_reverse(windows, window_size: Tuple[int, int], img_size):
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class MainHDR_gen(nn.Module):

    def __init__(self, dim=64, window_size=[4,4], num_heads=4):

        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.feature_extraction = ResBlock(in_channels=3, out_channels=dim, res_scale=1.0)
        # self.LayerNorm = nn.LayerNorm(dim)
        self.Norm = nn.GroupNorm(1, dim)
        # self.v = nn.Linear(in_features=dim, out_features=dim, bias=True) # v # input: num_windows*B L C
        # self.qk = nn.Linear(in_features=dim, out_features=dim*2, bias=True) # qk
        self.kv = nn.Linear(in_features=dim, out_features=dim*2, bias=True) # qk
        self.q = nn.Linear(in_features=dim, out_features=dim, bias=True) # qk
        attn_drop=0.0
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        proj_drop=0.0
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.FFN = GDFN(in_features=dim, hidden_features=dim*2)
        self.conv_last = Reconstruct_net(dim=dim, out_channels=3)

    def forward(self, ldr, sdr_out):

        f_sdr = self.feature_extraction(sdr_out) # B, dim, H, W 
        f_ldr = self.feature_extraction(ldr)
        

        B, C, H, W = f_sdr.shape # C=dim

        # f_sdr = self.LayerNorm(f_sdr.view(-1, C))
        # f_ldr = self.LayerNorm(f_ldr.view(-1, C))
        f_sdr = self.Norm(f_sdr)
        f_ldr = self.Norm(f_ldr)

        f_sdr = window_partition(f_sdr.permute(0, 2, 3, 1), self.window_size) # nW*B, wH, wW, C
        f_sdr = f_sdr.view(-1, prod(self.window_size), C) # nW*B, wH*wW, C
        f_ldr = window_partition(f_ldr.permute(0, 2, 3, 1), self.window_size) # nW*B, wH, wW, C
        f_ldr = f_ldr.view(-1, prod(self.window_size), C) # nW*B, wH*wW, C

        B_, N, C = f_sdr.shape # B_ = B*H*W/winsize/wimsize, N = winsize*winsize

        # qk = self.qk(f_ldr).reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4) # B_, 2N, C --> 2, B_, numheads, N, C/numheads
        # # q = qk[0]
        # # k = qk[1]
        # q, k = qk.unbind(0) 
        # v = self.v(f_sdr).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        kv = self.kv(f_sdr).reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4) # B_, 2N, C --> 2, B_, numheads, N, C/numheads
        k, v = kv.unbind(0) 
        q = self.q(f_ldr).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x) # B_, N, C

        x = f_sdr + x

        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        # x = window_reverse(x, self.window_size, (H, W)).view(B, C, H, W)
        x = window_reverse(x, self.window_size, (H, W)).permute(0, 3, 1, 2)
        x = self.FFN(x)
        x = self.conv_last(x)
        return x

class GDFN(nn.Module):
    def __init__(
        self,
        in_features=64,
        hidden_features=128,
        act_layer=nn.GELU
    ):
        super().__init__()
        self.out_features = in_features
        self.hidden_features = hidden_features
        # self.LayerNorm = nn.LayerNorm(in_features)
        self.Norm = nn.GroupNorm(1, in_features)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1)
        # self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1)
        self.act = act_layer()
        # self.fc3 = nn.Linear(hidden_features, self.out_features)
        self.fc3 = nn.Conv2d(hidden_features, self.out_features, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_copy = x.clone()
        # x = self.LayerNorm(x.view(-1, C)).view(B, C, H, W)
        x = self.Norm(x)
        x_1 = self.conv1(self.fc1(x))
        x_2 = self.conv2(self.fc2(x))
        x = self.fc3(x_1 * self.act(x_2))
        x = x_copy + x
        return x # -1, C

class Reconstruct_net(nn.Module):
    def __init__(
        self,
        dim=64,
        out_channels=3,
        conv_type="3conv"
    ):
        super().__init__()

        if conv_type == "1conv":
            self.block = nn.Conv2d(dim, dim, 3, 1, 1) 
        if conv_type == "3conv":
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim//2, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim//2, dim//2, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim//2, dim, 3, 1, 1),
            )
        self.conv_last = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.block(x)
        x = self.conv_last(x)
        return x


if __name__ == "__main__":
    
    sdr = torch.randn(1, 3, 512, 512)
    ldr = torch.randn(1, 3, 512, 512)
    model = MainHDR_gen(dim=64, window_size=[4,4], num_heads=4)
    output = model(ldr, sdr)
    print(output.shape)