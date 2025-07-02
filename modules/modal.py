import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import torchvision.transforms as transforms
from R2Gen.modules.visual_extractor import VisualExtractor
from R2Gen.modules.base_cmn import BaseCMN
from torch.nn import Parameter
from R2Gen.modules.Base import RGCN,ClassAttentionMechanism,DynamicGraphAttension,KeyEventAttention,PositionalEncoding,ImageCaptioningModule
import torch
from R2Gen.modules.weighted import WeightedContrastiveLoss
from R2Gen.modules.AFRM import AEMF



class MemoryDrivenAlignmentModule(nn.Module):
    def __init__(self, dim_in=2048, dim_hidden=612, memory_size=1000, num_heads=3, memory_update_rate=0.5, use_attention_bias=True):
        super(MemoryDrivenAlignmentModule, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = dim_hidden
        self.memory_update_rate = memory_update_rate
        self.use_attention_bias = use_attention_bias
        self.memory = nn.Parameter(torch.randn(memory_size, dim_hidden))       
        self.W_q = nn.Linear(dim_in, dim_hidden)
        self.W_k = nn.Linear(dim_hidden, dim_hidden)
        self.W_v = nn.Linear(dim_hidden, dim_hidden)
        self.out_proj = nn.Linear(dim_hidden * num_heads, dim_hidden) 
        self.memory_update = nn.Parameter(torch.randn(memory_size, dim_hidden)) 

        if use_attention_bias:
            self.attention_bias = nn.Parameter(torch.randn(1))

    def forward(self, visual_embeddings, positional_embeddings):
        b, num_patches, _ = visual_embeddings.shape
        query = self.W_q(visual_embeddings + positional_embeddings)         
        memory_keys = self.W_k(self.memory)
        memory_values = self.W_v(self.memory)

      
        head_outputs = []
        for _ in range(self.num_heads):
            scores = torch.matmul(query, memory_keys.T) / (self.hidden_dim ** 0.5)
            if self.use_attention_bias:
                scores += self.attention_bias 
            attention_weights = F.softmax(scores, dim=-1)
            response = torch.matmul(attention_weights, memory_values)
            head_outputs.append(response)

        
        combined_response = torch.cat(head_outputs, dim=-1)
        memory_update = self.memory_update * torch.mean(attention_weights, dim=1).unsqueeze(1) 
        updated_memory = self.memory + memory_update
        aligned_features = self.out_proj(combined_response)      
        self.memory.data = (1 - self.memory_update_rate) * self.memory.data + self.memory_update_rate * updated_memory.data 

        return aligned_features


class FeatureMemoryMappingModule(nn.Module):
    def __init__(self, dim_in, dim_hidden, memory_size=1000, memory_update_rate=0.5):

        super(FeatureMemoryMappingModule, self).__init__()
        self.dim_hidden = dim_hidden
        self.memory_update_rate = memory_update_rate
        self.memory = nn.Parameter(torch.randn(memory_size, dim_hidden))
        self.projection = nn.Linear(dim_in, dim_hidden) 
        self.memory_update_projection = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, visual_embeddings):

        projected_features = self.projection(visual_embeddings)
        distances = torch.cdist(projected_features, self.memory)
        closest_indices = torch.argmin(distances, dim=1)
        closest_memory_elements = self.memory[closest_indices]
        memory_update = self.memory_update_projection(projected_features - closest_memory_elements)
        updated_memory = self.memory.clone() 
        for i, index in enumerate(closest_indices):
            updated_memory[index] = (1 - self.memory_update_rate) * updated_memory[index] + \
                                    self.memory_update_rate * (updated_memory[index] + memory_update[i])
        self.memory.data = updated_memory

        return closest_memory_elements


