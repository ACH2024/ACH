from .rnn_agent import RNNAgent
from .P_Agent import P_Agent
from .masia_agent import MASIAAgent
from .maic_agent import MAICAgent


REGISTRY = {
    "rnn": RNNAgent,
    "P_Agent": P_Agent,
    "masia": MASIAAgent,
    "maic": MAICAgent,
}
