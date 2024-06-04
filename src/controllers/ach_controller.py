from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import time


# This multi-agent controller shares parameters between agents
class AchMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        if args.env == "grid":
            self.n_predators = args.env_args['num_adversaries']
            self.n_searchers = args.env_args['num_searchers']
        else:
            self.n_predators = args.n_p
            self.n_searchers = args.n_o
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.s_hidden_states = None
        self.p_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_inputs = agent_inputs.reshape(ep_batch.batch_size, self.n_agents, -1)
        p_inputs = agent_inputs[:, :self.n_predators, :].reshape(-1, self.input_shape)
        s_inputs = agent_inputs[:, self.n_predators:, :].reshape(-1, self.input_shape)

        s_hidden = self.s_agent.get_hidden(s_inputs, self.s_hidden_states).clone().detach()
        s_hidden = s_hidden.reshape(ep_batch.batch_size, self.n_searchers, -1)

        p_agent_outs, self.p_hidden_states, alpha = self.p_agent(p_inputs, self.p_hidden_states, s_hidden,
                                                                 agent_inputs, ep_batch.batch_size)
        s_agent_outs, self.s_hidden_states = self.s_agent(s_inputs, self.s_hidden_states)

        p_agent_outs = p_agent_outs.reshape(ep_batch.batch_size, self.n_predators, -1)
        s_agent_outs = s_agent_outs.reshape(ep_batch.batch_size, self.n_searchers, -1)
        agent_outs = th.cat((p_agent_outs, s_agent_outs), dim=1).reshape(-1, self.args.n_actions)

        loss_att = 0
        for b in range(ep_batch.batch_size):
            p_pi = th.softmax(p_agent_outs[b], dim=-1)
            s_pi = th.softmax(s_agent_outs[b], dim=-1)
            for i in range(self.n_predators):
                for j in range(self.n_searchers):
                    loss_kl = th.nn.functional.kl_div(p_pi[i].log(), s_pi[j], reduction="sum")
                    loss_att += alpha[b][i][j] * loss_kl

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), loss_att

    def init_hidden(self, batch_size):
        self.s_hidden_states = self.s_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_searchers, -1)  # bav
        self.p_hidden_states = self.p_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_predators, -1)  # bav

    def p_parameters(self):
        return self.p_agent.parameters()

    def s_parameters(self):
        return self.s_agent.parameters()

    def s_named_parameters(self):
        return self.s_agent.named_parameters()

    def p_named_parameters(self):
        return self.p_agent.named_parameters()

    def load_state(self, other_mac):
        self.p_agent.load_state_dict(other_mac.p_agent.state_dict())
        self.s_agent.load_state_dict(other_mac.s_agent.state_dict())

    def cuda(self):
        self.p_agent.cuda()
        self.s_agent.cuda()

    def save_models(self, path):
        th.save(self.p_agent.state_dict(), "{}/p_agent.th".format(path))
        th.save(self.s_agent.state_dict(), "{}/s_agent.th".format(path))

    def load_models(self, path):
        self.p_agent.load_state_dict(th.load("{}/p_agent.th".format(path), map_location=lambda storage, loc: storage))
        self.s_agent.load_state_dict(th.load("{}/s_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.p_agent = agent_REGISTRY[self.args.p_agent](input_shape, self.args)
        self.s_agent = agent_REGISTRY[self.args.s_agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
