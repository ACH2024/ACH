import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_distance_matrix(predator_positions, searcher_positions):
    """
    计算距离矩阵
    predator_positions: (N, 2) tensor, 其中 N 是 predator 的数量
    searcher_positions: (M, 2) tensor, 其中 M 是 searcher 的数量
    返回: (N, M) tensor, 其中每个元素 (i, j) 是 predator i 和 searcher j 之间的距离
    """
    diff = predator_positions.unsqueeze(1) - searcher_positions.unsqueeze(0)  # shape: (N, M, 2)
    dist = torch.sqrt(torch.sum(diff**2, dim=-1))  # shape: (N, M)
    return dist


def create_mask(distances, threshold):
    """
    根据距离阈值创建 mask
    distances: (N, M) tensor, 其中每个元素 (i, j) 是 predator i 和 searcher j 之间的距离
    threshold: scalar, 距离阈值
    返回: (N, M) tensor, 其中每个元素 (i, j) 是一个布尔值，如果 predator i 和 searcher j 之间的距离大于阈值，则为 True，否则为 False
    """
    mask = distances > threshold
    return mask


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.args = args
        self.input_dim = args.rnn_hidden_dim
        self.out_dim = args.rnn_hidden_dim
        self.hidden_dim = args.att_hidden_dim

        self.query = nn.Linear(self.input_dim, self.hidden_dim)
        self.key = nn.Linear(self.input_dim, self.hidden_dim)
        self.value = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, obs, predator_features, searcher_features, batch_size):
        """
        predator_features: (N, feature_dim) tensor, 其中 N 是 predator 的数量
        searcher_features: (M, feature_dim) tensor, 其中 M 是 searcher 的数量
        mask: (N, M) tensor, 其中每个元素 (i, j) 是一个布尔值，如果 predator i 和 searcher j 之间的距离大于阈值，则为 True，否则为 False
        返回: (N, feature_dim) tensor, 每个 predator 的新特征
        """
        n_p = self.args.env_args['num_adversaries']
        n_s = self.args.env_args['num_searchers']
        predator_features = predator_features.reshape(batch_size, n_p, -1)
        p_pos = obs[:, :n_p, 0:2]
        s_pos = obs[:, n_p:, 0:2]
        out_features = []
        out_weights = []

        for b in range(batch_size):
            dis = calculate_distance_matrix(p_pos[b], s_pos[b])
            mask = create_mask(dis, self.args.env_args['searcher_comm_range'])

            Q = self.query(predator_features[b])  # shape: (N, feature_dim)
            K = self.key(searcher_features[b])  # shape: (M, feature_dim)
            V = self.value(searcher_features[b])  # shape: (M, feature_dim)

            # 计算注意力分数
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.input_dim)  # shape: (N, M)
            scores = scores.masked_fill(mask, float('-inf'))  # 使用 mask

            # 计算注意力权重
            weights = F.softmax(scores, dim=-1)  # shape: (N, M)

            # 检查是否有 NaN
            nan_mask = torch.isnan(weights)
            if nan_mask.any():
                weights[nan_mask] = 0.0  # 将 NaN 设置为 0

            # 计算新的特征
            new_features = weights @ V  # shape: (N, feature_dim)

            new_features = self.fc(new_features)

            # 如果一个 predator 的距离范围内没有 searcher，使用其原始特征
            no_searcher_mask = weights.sum(dim=-1) == 0  # shape: (N,)
            new_features[no_searcher_mask] = predator_features[b][no_searcher_mask]

            out_features.append(new_features)
            out_weights.append(weights)

        out_features = torch.stack(out_features, dim=0)
        out_weights = torch.stack(out_weights, dim=0)
        return out_features, out_weights
