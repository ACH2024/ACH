import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossAtt(nn.Module):
    def __init__(self, args):
        super(CrossAtt, self).__init__()
        self.args = args

        self.in_features = args.rnn_hidden_dim
        self.out_features = args.rnn_hidden_dim
        self.hidden_size = args.att_hidden_dim

        self.Wq = nn.Parameter(torch.empty(size=(self.in_features, self.hidden_size)))
        nn.init.xavier_uniform_(self.Wq.data)
        self.Wk = nn.Parameter(torch.empty(size=(self.in_features, self.hidden_size)))
        nn.init.xavier_uniform_(self.Wk.data)
        self.Wv = nn.Parameter(torch.empty(size=(self.in_features, self.hidden_size)))
        nn.init.xavier_uniform_(self.Wv.data)

        self.fc = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, obs, p_hidden, s_hidden, batch_size):
        n_p = self.args.env_args['num_adversaries']
        n_s = self.args.env_args['num_searchers']

        p_hidden = p_hidden.reshape(batch_size, n_p, -1)

        p_pos = obs[:, :n_p, 0:2]
        s_pos = obs[:, n_p:, 0:2]

        obs_mask = torch.zeros((batch_size, n_p, n_s))
        for i in range(batch_size):
            for n in range(n_p):
                for m in range(n_s):
                    if (max(abs(p_pos[i][n] - s_pos[i][m])) - self.args.env_args['searcher_comm_range']) <= 0:
                        obs_mask[i][n][m] = 1
        h_out = []
        alpha_out = list()
        for b in range(batch_size):
            h_p = []
            alpha_p = []
            for i in range(n_p):
                h_ps = torch.cat((torch.unsqueeze(p_hidden[b][i], dim=0), s_hidden[b]), dim=0)
                adj = torch.zeros(1, (n_s + 1))
                adj[0][0] = 1
                s_index = []
                for j in range(n_s):
                    if obs_mask[b][i][j] == 1:
                        tmp_adj = torch.zeros(1, (n_s + 1))
                        tmp_adj[0][j + 1] = 1
                        adj = torch.cat((adj, tmp_adj), dim=0)
                        s_index.append(j)
                h = torch.matmul(adj, h_ps)
                if h.shape[0] == 1:
                    h_p.append(h)
                    alpha = torch.zeros(1, n_s)
                    alpha_p.append(alpha)
                else:
                    q = torch.matmul(h[:1, :], self.Wq)
                    k = torch.matmul(h[1:, :], self.Wk)
                    v = torch.matmul(h[1:, :], self.Wv)
                    e = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.hidden_size)
                    a = F.softmax(e, dim=-1)
                    alpha = torch.zeros(1, n_s)
                    index = 0
                    for s in s_index:
                        alpha[0][s] = a[0][index]
                        index += 1
                    alpha_p.append(alpha)
                    h_p.append(torch.matmul(a, v))
            h_p = torch.stack(h_p, dim=1)
            alpha_p = torch.stack(alpha_p, dim=1)
            h_out.append(h_p)
            alpha_out.append(alpha_p)
        alpha_out = torch.squeeze(torch.stack(alpha_out, dim=1))
        h_out = torch.squeeze(torch.stack(h_out, dim=1))

        h_out = self.fc(h_out)
        return h_out, alpha_out.reshape(batch_size, n_p, n_s)
