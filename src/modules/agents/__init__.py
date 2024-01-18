REGISTRY = {}

from .rnn_agent import RNNAgent
from .P_Agent import P_Agent
REGISTRY["rnn"] = RNNAgent
REGISTRY["sc"] = P_Agent