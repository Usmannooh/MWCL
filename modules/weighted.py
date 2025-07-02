import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np





class WeightedContrastiveLoss(nn.Module):
    class BaseOperator:
        def __init__(self, outer):
            self.outer = outer

        def operate(self, *args, **kwargs):
            raise NotImplementedError

    class ClusterRelationMaskCreator(BaseOperator):
        def operate(self, distance_matrix):
            with torch.set_grad_enabled(False):
                batch_size = distance_matrix.size(0)
                distance_matrix = distance_matrix - torch.eye(batch_size, batch_size, device='cuda') * 2
                row_indices = torch.arange(batch_size, device='cuda').unsqueeze(1).repeat(1, 1).flatten()
                col_indices = torch.topk(distance_matrix, 1, dim=1, sorted=False)[1].flatten()
                x_coords = torch.cat([row_indices, col_indices]).cpu().numpy()
                y_coords = torch.cat([col_indices, row_indices]).cpu().numpy()
                ones_arr = np.ones(x_coords.shape[0])
                graph = csr_matrix((ones_arr, (x_coords, y_coords)), shape=(batch_size, batch_size))
                _, cluster_labels = connected_components(csgraph=graph, directed=False, return_labels=True)
                cluster_labels = torch.tensor(cluster_labels, device='cuda')
                relation_mask = torch.eq(cluster_labels.unsqueeze(1), cluster_labels.unsqueeze(1).T)
            return relation_mask

    class LossEvaluator(BaseOperator):
        def operate(self, logit_vals, relation_mask, diag_block=None):
            if diag_block is not None:
                diag_block = 1 - diag_block
                relation_mask = relation_mask * diag_block
                exp_logit_vals = torch.exp(logit_vals) * diag_block
            else:
                exp_logit_vals = torch.exp(logit_vals)
            log_probs = logit_vals - torch.log(exp_logit_vals.sum(1, keepdim=True))
            mean_pos_log_prob = (relation_mask * log_probs).sum(1) / relation_mask.sum(1)
            loss_val = (-mean_pos_log_prob).mean()
            return loss_val

    def __init__(self, hidden_size=2048, output_size=2048):
        super().__init__()
        self.first_proj = FeatureProjection(in_dim=2048, out_dim=2048, hidden_dim=2048)
        self.second_proj = FeatureProjection(in_dim=2048, out_dim=2048, hidden_dim=2048)
        self.mask_creator = self.ClusterRelationMaskCreator(self)
        self.loss_evaluator = self.LossEvaluator(self)

    def forward(self, input_one, input_two, temp=0.1):  # 0.1
        ws = 1
        rank = 0
        batch_size = input_one.size(0)
        feature_one = F.normalize(self.first_proj(input_one))
        feature_two = F.normalize(self.first_proj(input_two))
        other_feature_one = get(feature_one)
        other_feature_two = get(feature_two)
        similarity = torch.cat([feature_one, feature_two]) @ torch.cat(
            [feature_one, feature_two, other_feature_one, other_feature_two]).T / temp
        diag_bool_mask = (1 - torch.eye(similarity.size(0), similarity.size(1), device='cuda')).bool()
        masked_logits = torch.masked_select(similarity, diag_bool_mask).reshape(similarity.size(0), -1)

        first_label_part = torch.arange(batch_size - 1, 2 * batch_size - 1).long().cuda()
        second_label_part = torch.arange(0, batch_size).long().cuda()
        combined_labels = torch.cat([first_label_part, second_label_part])
        feature_one_new = F.normalize(self.second_proj(input_one))
        feature_two_new = F.normalize(self.second_proj(input_two))
        total_batch_size = all_feature_one.size(0)

        relation_mask_one_list = []
        relation_mask_two_list = []
        if rank == 0:
            relation_mask_one = self.mask_creator.operate(all_feature_one @ all_feature_one.T).float()
            relation_mask_two = self.mask_creator.operate(all_feature_two @ all_feature_two.T).float()
            relation_mask_one_list = list(torch.chunk(relation_mask_one, ws))
            relation_mask_two_list = list(torch.chunk(relation_mask_two, ws))
            relation_mask_one = relation_mask_one_list[0]
            relation_mask_two = relation_mask_two_list[0]
        else:
            relation_mask_one = torch.zeros(batch_size, total_batch_size, device='cuda')
            relation_mask_two = torch.zeros(batch_size, total_batch_size, device='cuda')

        relation_mask_one_list = [relation_mask_one]
        relation_mask_two_list = [relation_mask_two]
        diag_block = torch.eye(total_batch_size, total_batch_size, device='cuda')
        diag_block = torch.chunk(diag_block, wws)[rank]
        weighted_loss = self.loss_evaluator.operate(feature_one_new @ all_feature_one.T / temp, relation_mask_two,
                                                    diag_block)
        weighted_loss += self.loss_evaluator.operate(feature_two_new @ all_feature_two.T / temp, relation_mask_one,
                                                     diag_block)
        weighted_loss /= 2

        return weighted_loss

class FeatureProjection(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048, hidden_dim=2048):
        super().__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.first_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.first_relu = nn.ReLU(True)
        self.second_linear = nn.Linear(hidden_dim, hidden_dim)
        self.second_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.second_relu = nn.ReLU(True)
        self.third_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, input_tensor):
        input_tensor = self.first_linear(input_tensor)
        input_tensor = self.first_batch_norm(input_tensor)
        input_tensor = self.first_relu(input_tensor)
        input_tensor = self.second_linear(input_tensor)
        input_tensor = self.second_batch_norm(input_tensor)
        input_tensor = self.second_relu(input_tensor)
        input_tensor = self.third_linear(input_tensor)
        return input_tensor

