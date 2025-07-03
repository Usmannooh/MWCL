import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanismBuilder:
    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction):
            super().__init__()
            self.op_seq = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(channels, channels // reduction, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, kernel_size=1)
            )

        def forward(self, inp):
            return self.op_seq(inp).sigmoid()

    class SpatialFeatureEnhancment(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)

        def forward(self, inp):
            return self.conv(inp)


class AFRM(nn.Module):
    def __init__(self, red=16):
        super().__init__()
        self.red_val = red
        in_chan = 2048
        feat_chan = 2048
        self.conv_start_1 = nn.Conv2d(in_chan, feat_chan, kernel_size=3, padding=1)
        self.conv_start_2 = nn.Conv2d(feat_chan, feat_chan, kernel_size=3, padding=1)
        self.chan_attn = AttentionMechanismBuilder.ChannelAttention(feat_chan, red)
        self.spat_attn = AttentionMechanismBuilder.SpatialFeatureEnhancment(feat_chan)

    def forward(self, inp_tensor):
        proc_tensor = F.relu(self.conv_start_1(inp_tensor))
        proc_tensor = self.conv_start_2(proc_tensor)
        chan_map = self.chan_attn(proc_tensor)
        spat_map = self.spat_attn(proc_tensor)
        comb_map = torch.add(chan_map, spat_map)
        attn_applied = torch.mul(proc_tensor, comb_map)
        final_out = torch.add(inp_tensor, attn_applied)
        return final_out


class AEMF(nn.Module): #Attention-Enhanced Multi-Modal Fusion (AEMF)
    def __init__(self, visual_dim, other_dim, hidden_dim, red=4):
        super().__init__()
        self.enhancer = AFRM(red)
        self.fc_first = nn.Linear(visual_dim + other_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc_second = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vis_feats, other_feats):
        improved_vis = self.enhancer(vis_feats)
        merged = torch.cat((improved_vis, other_feats), dim=-1)
        x = self.fc_first(merged)
        x = self.activation(x)
        x = self.fc_second(x)
        return x

