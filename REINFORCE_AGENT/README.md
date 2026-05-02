# REINFORCE_AGENT

## Agent Info
- Agent name: `REINFORCE_AGENT`
- Team: `reinforce`
- Authors:
  - Chengyin Xu (`cxu371@gatech.edu`)
  - Mohit Talreja (`mtalreja6@gatech.edu`)
  - Ethan Mendes (`emendes3@gatech.edu`)

## What This Submission Contains
This folder is the final agent submitted to Gradescope. The inference code restores a PPO checkpoint and outputs SoccerTwos branched actions through `ActionFlattener`.

## File Structure (Final Agent)
- `agent.py`: Inference loader (`RayPPOAgent`) and action mapping logic.
- `__init__.py`: Agent package entry point.
- `checkpoint-1075` and `checkpoint-1075.tune_metadata`: Final trained policy checkpoint.
- `checkpoint_path.txt`: Relative path used by `agent.py` to restore the checkpoint.

## Modification Summary (for Project Rubric)
Our final approach uses **PPO + player-level self-play + dense reward shaping**.

### 1) Reward Modification (Implemented)
Reward shaping is implemented in:
- `utils.py` (`RewardShapingWrapper`)

Shaped reward terms used in final run:
- `reward_scale = 1.0`
- `living_reward = 0.0`
- `dense_distance_coef = 0.01`
- `touch_bonus = 0.01`
- `touch_threshold = 1.25`

Dense shaping logic:
- Positive bonus for reducing distance to the ball.
- Additional touch bonus when entering near-ball range (crossing threshold).

### 2) Self-Play Setup (Learning Regime)
Self-play training script:
- `train_ppo_cpu_option2_selfplay_dense.py`

Batch job script used on PACE:
- `scripts/soccer_option2_selfplay_dense.batch`

Self-play design:
- Multi-policy pool: `default`, `opponent_1`, `opponent_2`, `opponent_3`
- Train only `default`
- Periodic opponent updates from `default` (league-style archive updates)

## Training Details (Final Run)
- Algorithm: PPO (Ray RLlib, Torch backend)
- Environment variation: `EnvType.multiagent_player`
- CPUs: 12 requested in Slurm (`num_workers=8`, `num_envs_per_worker=1`)
- GPUs: 0
- Rollout fragment length: 500
- Train batch size: 4000
- Stop config: time limit 8h and target timesteps 15M (run stopped by wall-time)

Final completed run summary:
- `timesteps_total = 4,308,000`
- `agent_timesteps_total = 17,232,000`
- Final checkpoint: `checkpoint-1075`

## High-Level Results
Final Gradescope performance (this agent):
- vs Random: 10/10 wins
- vs CEIA Baseline: 8/10 wins
- vs TA Agent: strong competitive performance (several wins; extra credit partial)

## Reward Curve Artifact for Report
Submission-2-only curve (`policy_reward_mean/default` vs steps):
- `report_artifacts/reward_curve_submission2_policy_default.svg`
- `report_artifacts/reward_curve_submission2_policy_default.png`

Plot generation script:
- `scripts/plot_reward_curves.py`
