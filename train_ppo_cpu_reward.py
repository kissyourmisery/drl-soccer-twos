import argparse

import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only PPO training with optional reward shaping."
    )
    parser.add_argument("--exp-name", type=str, default="PPO_reward_cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--timesteps-total", type=int, default=20000000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--living-reward", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name=args.exp_name,
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": args.num_envs_per_worker,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "reward_scale": args.reward_scale,
                "living_reward": args.living_reward,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
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
