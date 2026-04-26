# REINFORCE_AGENT

## Agent Info

- Agent name: `REINFORCE_AGENT`
- Team: `reinforce`
- Authors:
  - Chengyin Xu (`cxu371@gatech.edu`)
  - Mohit Talreja (`mtalreja6@gatech.edu`)
  - Ethan Mendes (`emendes3@gatech.edu`)

## Description

This agent restores a PPO checkpoint trained in SoccerTwos with player-level
self-play plus mild dense reward shaping in `utils.py`:
- distance-to-ball shaping (`dense_distance_coef`)
- touch bonus near the ball (`touch_bonus`, `touch_threshold`)

At inference time it maps PPO discrete actions to continuous branched actions
using `ActionFlattener`.

## Runtime Notes

- Inference checkpoint path is set in `checkpoint_path.txt`.
- For this submission, the checkpoint file is bundled inside this folder.
