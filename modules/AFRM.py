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













import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
# class RAM(nn.Module):
#     def __init__(self, reduction=16):
#         super(RAM, self).__init__()
#         self.reduction = reduction
#         in_channels = 2048
#         channel = 2048
#
#         # Initial convolution layers
#         self.conv1 = nn.Conv2d(in_channels, channel, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
#
#         # Channel attention
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(channel, channel // reduction, kernel_size=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, kernel_size=1)
#         )
#
#         # Spatial attention
#         self.spatial_attention = nn.Conv2d(channel, 1, kernel_size=3, padding=1, bias=False)
#
#     def forward(self, input):
#         u = F.relu(self.conv1(input))
#         u = self.conv2(u)
#         x = self.channel_attention(u).sigmoid()# Channel attention
#         y = self.spatial_attention(u)# Spatial attention
#         z = torch.add(x, y)# Combine attention maps
#         z = torch.mul(u, z)
#         z = torch.add(input, z)
#
#         return z
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class RAM(nn.Module):
#     def __init__(self, reduction=16, channel_activation='gelu', attn_map_activation='softmax'):
#         super(RAM, self).__init__()
#         self.reduction_factor = reduction
#         input_channels = 2048
#         feature_channels = 2048
#
#         # Initial convolutional layers
#         self.initial_conv_1 = nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1)
#         self.initial_conv_2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
#
#         # Channel - based attention mechanism
#         channel_activation_layer = self.get_activation(channel_activation)
#         self.channel_aware_attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(feature_channels, feature_channels // reduction, kernel_size=1),
#             channel_activation_layer,
#             nn.Conv2d(feature_channels // reduction, feature_channels, kernel_size=1)
#         )
#         # Spatial - based attention mechanism
#         self.spatial_aware_attn = nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1, bias=False)
#         self.attn_map_activation = self.get_activation(attn_map_activation)
#
#     def get_activation(self, activation_name):
#
#         if activation_name == 'swish':
#             return nn.SiLU()
#         elif activation_name == 'gelu':
#             return nn.GELU()
#         elif activation_name == 'relu':
#             return nn.ReLU()
#         elif activation_name == 'softmax':
#             return nn.Softmax(dim = 1)
#         elif activation_name == 'tanh':
#             return nn.Tanh()
#         elif activation_name == 'sigmoid':
#             return nn.Sigmoid()
#         else:
#             raise ValueError(f"Unsupported activation function: {activation_name}")
#
#     def forward(self, input_tensor):
#         # Pass through initial convolutions
#         processed_tensor = F.relu(self.initial_conv_1(input_tensor))
#         processed_tensor = self.initial_conv_2(processed_tensor)
#         channel_attn_map = self.channel_aware_attn(processed_tensor)# Compute channel attention
#         channel_attn_map = self.attn_map_activation(channel_attn_map)
#         spatial_attn_map = self.spatial_aware_attn(processed_tensor)# Compute spatial attention
#         combined_attn_map = torch.add(channel_attn_map, spatial_attn_map) # Combine attention maps
#         attended_tensor = torch.mul(processed_tensor, combined_attn_map)# Apply combined attention to the processed tensor
#         output_tensor = torch.add(input_tensor, attended_tensor) # Residual connection
#
#         return output_tensor
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class RAM(nn.Module):
#     def __init__(self, reduction=16):
#         super(RAM, self).__init__()
#         self.reduction_factor = reduction
#         input_channels = 2048
#         feature_channels = 2048
#
#         # Initial convolutional layers
#         self.initial_conv_1 = nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1)
#         self.initial_conv_2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
#
#         # Channel - based attention mechanism
#         self.channel_aware_attn = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(feature_channels, feature_channels // reduction, kernel_size=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(feature_channels // reduction, feature_channels, kernel_size=1)        )
#         self.spatial_aware_attn = nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1, bias=False) # Spatial - based attention mechanism
#
#     def forward(self, input_tensor):
#         # Pass through initial convolutions
#         processed_tensor = F.relu(self.initial_conv_1(input_tensor))
#         processed_tensor = self.initial_conv_2(processed_tensor)
#         channel_attn_map = self.channel_aware_attn(processed_tensor).sigmoid()# Compute channel attention
#         spatial_attn_map = self.spatial_aware_attn(processed_tensor)# Compute spatial attention
#         combined_attn_map = torch.add(channel_attn_map, spatial_attn_map)# Combine attention maps
#         attended_tensor = torch.mul(processed_tensor, combined_attn_map) # Apply combined attention to the processed tensor
#         output_tensor = torch.add(input_tensor, attended_tensor) # Residual connection
#
#         return output_tensor


# class MultiModalFusionModule(nn.Module):
#     def __init__(self, visual_dim, other_dim, hidden_dim, reduction=4):
#         super(MultiModalFusionModule, self).__init__()
#
#         self.ram = RAM(reduction)# Initialize the RAM module for feature enhancement
#         # Fully connected layers for fusion
#         self.fc1 = nn.Linear(visual_dim + other_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#
#     def forward(self, visual_embeddings, other_embeddings):
#         enhanced_visual_features = self.ram(visual_embeddings)# Apply RAM to the visual embeddings for enhanced feature representation
#         combined = torch.cat((enhanced_visual_features, other_embeddings), dim=-1)# Concatenate the enhanced visual features with other embeddings
#
#         # Process the combined features through the fully connected layers
#         x = self.fc1(combined)
#         x = self.relu(x)
#         x = self.fc2(x)
#
#         return x
