import torch.nn as nn
import torch.nn.functional as F
from modules.sc_module.CrossAtt import CrossAtt


class P_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(P_Agent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        if args.sc_mode == "att":
            self.att = CrossAtt(args)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, s_hidden, obs_all, batch_size):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        att_out, alpha = self.att(obs_all, h, s_hidden, batch_size)

        h_att = h + att_out.reshape(-1, self.args.rnn_hidden_dim)
        q = self.fc2(h_att)
        return q, h, alpha

