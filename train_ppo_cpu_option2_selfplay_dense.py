import argparse
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env


UPDATE_THRESHOLD = 0.5


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id in (0, 1):
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.50, 0.25, 0.15, 0.10],
    )


class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        episode_reward_mean = info["result"].get("episode_reward_mean", float("-inf"))
        if episode_reward_mean > UPDATE_THRESHOLD:
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Option 2: PPO self-play (player-level) with mild dense shaping."
    )
    parser.add_argument("--exp-name", type=str, default="PPO_option2_selfplay_dense_cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--hours", type=float, default=7.0)
    parser.add_argument("--timesteps-total", type=int, default=20000000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--living-reward", type=float, default=0.0)
    parser.add_argument("--dense-distance-coef", type=float, default=0.015)
    parser.add_argument("--touch-bonus", type=float, default=0.01)
    parser.add_argument("--touch-threshold", type=float, default=1.25)
    parser.add_argument("--selfplay-threshold", type=float, default=0.5)
    parser.add_argument("--base-port", type=int, default=56039)
    parser.add_argument("--worker-id-offset", type=int, default=200)
    parser.add_argument("--ray-address", type=str, default="")
    parser.add_argument("--ray-temp-dir", type=str, default="")
    parser.add_argument("--checkpoint-freq", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Keep each run on its own local Ray runtime folder and disable dashboard
    # components that have been unstable on shared HPC nodes.
    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

    UPDATE_THRESHOLD = args.selfplay_threshold

    ray.shutdown()
    ray_address = args.ray_address.strip()
    if ray_address.lower() in ("", "none", "local"):
        ray_address = None

    ray_init_kwargs = {
        "address": ray_address,
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    ray_temp_dir = args.ray_temp_dir.strip()
    if ray_temp_dir:
        # Keep this path short enough for Ray's AF_UNIX socket length limits.
        ray_temp_dir = os.path.abspath(ray_temp_dir)
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray_init_kwargs["_temp_dir"] = ray_temp_dir

    ray.init(**ray_init_kwargs)
    tune.registry.register_env("Soccer", create_rllib_env)

    temp_env = create_rllib_env(
        {
            "variation": EnvType.multiagent_player,
            "flatten_branched": True,
            "base_port": args.base_port,
            "worker_id": args.worker_id_offset,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name=args.exp_name,
        config={
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayUpdateCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "variation": EnvType.multiagent_player,
                "flatten_branched": True,
                "base_port": args.base_port,
                "worker_id_offset": args.worker_id_offset,
                "reward_scale": args.reward_scale,
                "living_reward": args.living_reward,
                "dense_distance_coef": args.dense_distance_coef,
                "touch_bonus": args.touch_bonus,
                "touch_threshold": args.touch_threshold,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 4000,
        },
        stop={
            "timesteps_total": args.timesteps_total,
            "time_total_s": int(args.hours * 3600),
        },
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print("Best trial:", best_trial)
    print("Best checkpoint:", best_checkpoint)
    print("Done training")
