from .envs.predator_prey import PredatorPrey


def grid_Env(args):
    if args['scenario_name'] == "predator-prey":
        map_size = args['map_size']
        n_agents = args['num_agents']
        n_preys = args['num_preys']
        n_predators = args['num_adversaries']
        n_searchers = args['num_searchers']
        predators_obs_range = args['predators_obs_range']
        comm_range = args['searcher_comm_range']
        max_steps = args['episode_length']
        env = PredatorPrey(args=args, grid_shape=(map_size, map_size),
                           n_agents=n_agents, n_preys=n_preys, n_searchers=n_searchers, n_predators=n_predators,
                           prey_move_probs=(0, 0, 0, 0, 1), full_observable=False, penalty=-0.5,
                           step_cost=-0.1, prey_capture_reward=20, max_steps=max_steps,
                           agent_view_mask=(map_size, map_size), obs_range=predators_obs_range, comm_range=comm_range)
    else:
        print("Can not support the " + args['scenario_name'] + "scenario.")
        raise NotImplementedError
    return env
