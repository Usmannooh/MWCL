import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from R2Gen.modules.visual_extractor import VisualExtractor
from R2Gen.modules.base_cmn import BaseCMN
from torch.nn import Parameter
from R2Gen.module. import memory
from R2Gen.modules.memory import FeatureMemoryMappingModule,MemoryDrivenAlignmentModule






class Sman(nn.Module):
    def __init__(self, args, tokenizer, num_classes, forward_adj, backward_adj, feature_dim=2048, embed_size=256,
                 hidden_size=612):
        super(Sman, self).__init__()

        self.args = args
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.forward_adj = forward_adj
        self.backward_adj = backward_adj
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        



        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad == True, self.parameters())
        total_params = sum(np.prod(p.size()) for p in model_parameters)
        return f'{self.__class__.__name__}\nTrainable parameters: {total_params}'

    def init_hidden(self, batch_size):
        device = torch.device('cuda')
        hidden_state = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        return hidden_state, cell_state

    def forward(self, img_features, enc_features, captions, **kwargs):
        batch_size = img_features.size(0)


        img_features = self.positional_encoding(img_features)
        context_features, alpha = self.attention(enc_features, img_features)
        context_features = context_features.view(batch_size, self.feature_dim, 1, 1)
        visual_features = self.visual_extractor(img_features)
        visual_features = self.graph_cn(visual_features, forward_adj=self.normalized_forward_adj,
                                        backward_adj=self.normalized_backward_adj)
        visual_features = self.class_attention(visual_features)

        # Flatten visual features for captioning
        flattened_visual = visual_features.view(batch_size, -1)
        self.hidden = (self.init_sent_h(flattened_visual), self.init_sent_c(flattened_visual))
        output_captions = self.captioning(visual_features, captions)  # Image Captioning

        return output_captions, alpha

    def forward_iu_xray(self, images, captions=None, mode='train', update_opts={}):
        att_features = []
        func_features = []

        for i in range(2):
            att_feat, func_feat = self.visual_extractor(images[:, i])
            att_features.append(att_feat)
            func_features.append(func_feat)

        func_feature = torch.cat(func_features, dim=1)
        forward_adj = self.normalized_forward_adj.repeat(6, 1, 1)
        backward_adj = self.normalized_backward_adj.repeat(6, 1, 1)
        global_features = [feat.mean(dim=(2, 3)) for feat in att_features]
        att_features = [self.class_attention(feat, self.num_classes) for feat in att_features]

        for idx in range(2):
            att_features[idx] = torch.cat((global_features[idx].unsqueeze(1), att_features[idx]), dim=1)
            att_features[idx] = self.linear_trans_lyr_2(att_features[idx].transpose(1, 2)).transpose(1, 2)
        att_feature_combined = torch.cat(att_features, dim=1)
        att_feature_combined = self.linear_trans_lyr(att_feature_combined.transpose(1, 2)).transpose(1, 2)

        if mode == 'train':
            return self.encoder_decoder(func_feature, att_feature_combined, captions, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(func_feature, att_feature_combined, mode='sample', update_opts=update_opts)
        else:
            raise ValueError("Invalid mode provided.")

    def forward_mimic_cxr(self, images, captions=None, mode='train', update_opts={}):
        att_features, func_feature = self.visual_extractor(images)
        if mode == 'train':
            return self.encoder_decoder(func_feature, att_features, captions, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(func_feature, att_features, mode='sample', update_opts=update_opts)
        else:
            raise ValueError("Invalid mode provided.")
