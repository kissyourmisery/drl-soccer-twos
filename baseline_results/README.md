# Baseline Results (Submission 1 Recovery)

This folder stores recovered artifacts for the first training run so we can compare against the final self-play agent.

## Files

- `submission1_soccer_cpu_reward-5044331.out`
  - Full recovered Slurm log from commit `89e23d1`.
- `submission1_reward_timeseries.csv`
  - Extracted series used for plotting:
  - Columns: `timesteps_total`, `episode_reward_mean`
- `reward_curve_submission1_baseline.svg`
- `reward_curve_submission1_baseline.png`
  - Baseline reward curve rendered in the same style as the final Submission 2 plot.

## Data Source

- Recovered from: `soccer_cpu_reward-5044331.out` in historical commit `89e23d1`
- Extraction logic: parse each `Result for PPO_Soccer_876b4_00000` block and keep `(timesteps_total, episode_reward_mean)` entries.

## Training Setup for This Recovered Run

- PPO (`ray[rllib]`, CPU-only)
- Team-vs-policy, single-player mode
- No self-play league
- `reward_scale=1.15`, `living_reward=0.0`
- ~9.14M timesteps at termination
