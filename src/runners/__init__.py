REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_full import EpisodeRunner_full
REGISTRY["episode_full"] = EpisodeRunner_full

from .parallel_runner_x import ParallelRunner_x
REGISTRY["parallel_x"] = ParallelRunner_x

from .parallel_runner_ach import ParallelRunner_ach
REGISTRY["parallel_ach"] = ParallelRunner_ach

from .msra_episode_runner import EpisodeRunner
REGISTRY["msra_episode"] = EpisodeRunner

from .msra_parallel_runner import ParallelRunner
REGISTRY["msra_parallel"] = ParallelRunner

from .parallel_runner_pp import ParallelRunner
REGISTRY["parallel_pp"] = ParallelRunner
