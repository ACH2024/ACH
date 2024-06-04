from .envs.predator_prey import PredatorPrey
from .envs.traffic_junction import TrafficJunction


def ppEnv(num_searchers,
        num_predators,
        num_preys,
        map_size,
        predators_obs_range,
        searcher_comm_range,
        episode_length,
        prey_capture_reward,
        obs_last_action,
        state_last_action,
        print_rew,
        is_print,
        print_steps,
        seed):
    env = PredatorPrey(grid_shape=(map_size, map_size), n_agents=(num_searchers+num_predators), n_preys=num_preys,
                       n_searchers=num_searchers, n_predators=num_predators,prey_move_probs=(0, 0, 0, 0, 1),
                       full_observable=False, penalty=-0.5, step_cost=-0.1, prey_capture_reward=prey_capture_reward,
                       max_steps=episode_length, agent_view_mask=(map_size, map_size), obs_range=predators_obs_range,
                       comm_range=searcher_comm_range)
    return env


def tjEnv(num_agents,
          map_size,
          collision_reward,
          full_observable,
          max_steps,
          obs_last_action,
          state_last_action,
          print_rew,
          is_print,
          print_steps,
          seed,
          ):
    env = TrafficJunction(grid_shape=(map_size, map_size), step_cost=-0.01, n_max=num_agents,
                          collision_reward=collision_reward, arrive_prob=0.5, full_observable=full_observable,
                          max_steps=max_steps)
    return env
