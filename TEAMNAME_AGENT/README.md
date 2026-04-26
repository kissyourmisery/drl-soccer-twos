# Ray PPO Agent

`ray_ppo_agent` restores a PPO checkpoint at inference time and maps one-player
actions to SoccerTwos team actions for `watch` and autograder runs.

## Setup

1. Train with `train_ppo_cpu_reward.py` (single-player flattened policy).
2. Find a checkpoint file (for example: `ray_results/.../checkpoint_000005/checkpoint-5`).
3. Put that path in `checkpoint_path.txt`.
   - Use an absolute path while testing locally.
   - For zip submission, copy checkpoint files into this folder and use a relative path.

## Test

```bash
python -m soccer_twos.watch -m ray_ppo_agent
python -m soccer_twos.watch -m1 ray_ppo_agent -m2 example_team_agent
python -m soccer_twos.watch -m1 ray_ppo_agent -m2 ceia_baseline_agent
```
