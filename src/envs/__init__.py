from functools import partial
from smac.env import MultiAgentEnv
from smac_plus import StarCraft2Env, Tracker1Env, Join1Env
from .ma_gym.grid_Env import ppEnv, tjEnv
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
	return env(**kwargs)


REGISTRY = {
	"sc2": partial(env_fn, env=StarCraft2Env),
	"tracker1": partial(env_fn, env=Tracker1Env),
	"join1": partial(env_fn, env=Join1Env),
	"pp": partial(env_fn, env=ppEnv),
	"tj": partial(env_fn, env=tjEnv),
}

if sys.platform == "linux":
	os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
