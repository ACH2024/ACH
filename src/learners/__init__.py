from .q_learner import QLearner
from .coma_learner import COMALearner
from .categorical_q_learner import CateQLearner
from .ach_learner import AchLearner
from .masia_learner import MASIALearner
from .maic_learner import MAICLearner
from .qtran_learner import QLearner as QTRANLearner


REGISTRY = {
    "q_learner": QLearner,
    "coma_learner": COMALearner,
    "cate_q_learner": CateQLearner,
    "ach_learner": AchLearner,
    "masia_learner": MASIALearner,
    "maic_learner": MAICLearner,
    "qtran_learner": QTRANLearner,
}
