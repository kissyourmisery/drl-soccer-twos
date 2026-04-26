import os

import gym
from gym_unity.envs import ActionFlattener
import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface


class _DummySinglePlayerEnv(gym.Env):
    """
    Dummy env used only to bootstrap PPOTrainer at inference time.
    """

    def __init__(self, config=None):
        config = config or {}
        obs_size = int(config.get("obs_size", 336))
        action_size = int(config.get("action_size", 27))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(action_size)

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        return self.reset(), 0.0, True, {}


class RayPPOAgent(AgentInterface):
    """
    Inference-time agent that restores a PPO checkpoint trained with
    single-player + flattened branched actions.
    """

    _trainer = None

    def __init__(self, env):
        super().__init__()
        self.name = "RAY_PPO_AGENT"
        self.flattener = ActionFlattener(env.action_space.nvec)

        if RayPPOAgent._trainer is None:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, include_dashboard=False)

            register_env("DummySinglePlayerEnv", lambda cfg: _DummySinglePlayerEnv(cfg))

            agent_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path_file = os.path.join(agent_dir, "checkpoint_path.txt")
            if not os.path.isfile(checkpoint_path_file):
                raise FileNotFoundError(
                    "Missing checkpoint_path.txt in agent folder. "
                    "Write the checkpoint file path there."
                )

            with open(checkpoint_path_file, "r", encoding="utf-8") as f:
                checkpoint_path = f.read().strip()

            if not checkpoint_path:
                raise ValueError("checkpoint_path.txt is empty.")

            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(agent_dir, checkpoint_path)
                checkpoint_path = os.path.abspath(checkpoint_path)

            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found at: {checkpoint_path}"
                )

            trainer = PPOTrainer(
                env="DummySinglePlayerEnv",
                config={
                    "num_workers": 0,
                    "num_gpus": 0,
                    "framework": "torch",
                    "model": {
                        "vf_share_layers": True,
                        "fcnet_hiddens": [512, 512],
                    },
                    "env_config": {
                        "obs_size": env.observation_space.shape[0],
                        "action_size": int(np.prod(env.action_space.nvec)),
                    },
                },
            )
            trainer.restore(checkpoint_path)
            RayPPOAgent._trainer = trainer

        self.trainer = RayPPOAgent._trainer

    def act(self, observation):
        actions = {}
        for player_id, player_obs in observation.items():
            player_obs = np.asarray(player_obs, dtype=np.float32)
            if hasattr(self.trainer, "compute_single_action"):
                action_idx = self.trainer.compute_single_action(
                    player_obs, explore=False
                )
            else:
                action_idx = self.trainer.compute_action(player_obs, explore=False)
            actions[player_id] = self.flattener.lookup_action(int(action_idx))
        return actions
