from learning_and_planning.experiments.helpers.specification_helper import \
    create_experiments_helper

base_config = {"create_agent.agent_name": "@MCTSWithVoting",
               "create_agent.value_function_name": "ensemble",
               "create_agent.replay_capacity": 1000,

               "get_env_creator.env_callable_name": "gym_sokoban_fast:SokobanEnvFast",
               "SokobanEnvFast.dim_room": (10, 10),
               "SokobanEnvFast.num_boxes": 4,
               "SokobanEnvFast.mode": "one_hot",
               "SokobanEnvFast.penalty_for_step": 0.,
               "SokobanEnvFast.reward_box_on_target": 0.,
               "SokobanEnvFast.reward_finished": 10.,

               "ValueBase.optimizer_fn": "@tf.train.RMSPropOptimizer",
               "ValueBase.learning_rate_fn": 0.00025,
               "ValueBase.max_tf_checkpoints_to_keep": 10,
               "ValueBase.loss_fn": "@tf.losses.mean_squared_error",
               "ValueBase.loss_clip_threshold": 0.0,
               "ValueBase.activation": "identity",
               "ValueBase.decay": 0.,
               "ValueBase.model_name": "convnet_mnist",

               "ValueEnsemble.include_prior": False,
               "ValueEnsemble.accumulator_fn": "@EnsembleValueAccumulatorVoting",
               "EnsembleValueAccumulatorVoting.bonus_fn": "@ucb1",  # ucb1
               "EnsembleValueAccumulatorVoting.bonus_loading": 0.,
               "ValueEnsemble.num_ensemble": 3,

               "EnsembleValueTraits.dead_end_value": -2.0,
               "MCTSWithVoting.num_mcts_passes": 10,
               "MCTSWithVoting.num_sampling_moves": 0,
               "MCTSWithVoting.avoid_loops": True,
               "MCTSWithVoting.gamma": 0.99,
               "MCTSWithVoting.episode_max_steps": 200,
               "MCTSWithVoting.avoid_history_coeff": -2.,

               "Server.min_replay_history": 1000,
               "PoloOutOfGraphReplayBuffer.solved_unsolved_ratio": 0.5,
               "curriculum": False,
               "training_steps": 500000,
               "train_every_num_steps": 100,
               "game_buffer_size": 25,
               "run_eval_worker": False,
               "Server.log_every_n": 50,
               "MCTSWithVoting.node_value_mode": "bootstrap",
               }


params_grid = {
    "PoloWrappedReplayBuffer.batch_size": [96],
}

experiments_list = create_experiments_helper(experiment_name='sokoban_ensembles',
                                             python_path='.:./deps/gym-sokoban:./deps/ourlib:'
                                                         './deps/baselines:./deps/dopamine:./deps/gym-sokoban-fast:./deps/chainenv:./polo_plus/kc:./deps/toy-mr:',
                                             paths_to_dump='',
                                             base_config=base_config, params_grid=params_grid)
