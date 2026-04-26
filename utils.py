from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RewardShapingWrapper(gym.core.Wrapper):
    """
    Applies lightweight reward shaping on top of the base env reward.
    """

    def __init__(
        self,
        env,
        reward_scale=1.0,
        living_reward=0.0,
        dense_distance_coef=0.0,
        touch_bonus=0.0,
        touch_threshold=1.25,
    ):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.living_reward = living_reward
        self.dense_distance_coef = dense_distance_coef
        self.touch_bonus = touch_bonus
        self.touch_threshold = touch_threshold
        self._prev_dist_to_ball = {}

    def reset(self, **kwargs):
        self._prev_dist_to_ball.clear()
        return self.env.reset(**kwargs)

    def _dense_bonus_for_agent(self, agent_key, agent_info):
        if not isinstance(agent_info, dict):
            return 0.0

        player_info = agent_info.get("player_info")
        ball_info = agent_info.get("ball_info")
        if not isinstance(player_info, dict) or not isinstance(ball_info, dict):
            return 0.0

        player_pos = player_info.get("position")
        ball_pos = ball_info.get("position")
        if player_pos is None or ball_pos is None:
            return 0.0

        player_pos = np.asarray(player_pos, dtype=np.float32)
        ball_pos = np.asarray(ball_pos, dtype=np.float32)
        if player_pos.shape[0] < 2 or ball_pos.shape[0] < 2:
            return 0.0

        dist_to_ball = float(np.linalg.norm(player_pos[:2] - ball_pos[:2]))
        prev_dist = self._prev_dist_to_ball.get(agent_key)
        bonus = 0.0

        if prev_dist is not None and self.dense_distance_coef != 0.0:
            bonus += max(prev_dist - dist_to_ball, 0.0) * self.dense_distance_coef

        if (
            self.touch_bonus != 0.0
            and prev_dist is not None
            and prev_dist > self.touch_threshold
            and dist_to_ball <= self.touch_threshold
        ):
            bonus += self.touch_bonus

        self._prev_dist_to_ball[agent_key] = dist_to_ball
        return bonus

    def _dense_bonus(self, reward, info):
        if (
            self.dense_distance_coef == 0.0
            and self.touch_bonus == 0.0
        ):
            if isinstance(reward, dict):
                return {agent_id: 0.0 for agent_id in reward}
            return 0.0

        if isinstance(reward, dict):
            bonuses = {}
            for agent_id in reward:
                agent_info = info.get(agent_id) if isinstance(info, dict) else None
                bonuses[agent_id] = self._dense_bonus_for_agent(agent_id, agent_info)
            return bonuses

        scalar_info = info if isinstance(info, dict) else None
        return self._dense_bonus_for_agent("single", scalar_info)

    def _shape_reward(self, reward, info):
        dense_bonus = self._dense_bonus(reward, info)
        if isinstance(reward, dict):
            return {
                agent_id: (
                    (float(agent_reward) * self.reward_scale)
                    + self.living_reward
                    + float(dense_bonus.get(agent_id, 0.0))
                )
                for agent_id, agent_reward in reward.items()
            }
        return (
            (float(reward) * self.reward_scale)
            + self.living_reward
            + float(dense_bonus)
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, self._shape_reward(reward, info), done, info


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    raw_env_config = env_config
    env_config = dict(env_config)
    worker_id_offset = int(env_config.pop("worker_id_offset", 0))
    if hasattr(raw_env_config, "worker_index"):
        env_config["worker_id"] = (
            raw_env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + raw_env_config.vector_index
            + worker_id_offset
        )
    reward_scale = float(env_config.pop("reward_scale", 1.0))
    living_reward = float(env_config.pop("living_reward", 0.0))
    dense_distance_coef = float(env_config.pop("dense_distance_coef", 0.0))
    touch_bonus = float(env_config.pop("touch_bonus", 0.0))
    touch_threshold = float(env_config.pop("touch_threshold", 1.25))
    env = soccer_twos.make(**env_config)
    if (
        reward_scale != 1.0
        or living_reward != 0.0
        or dense_distance_coef != 0.0
        or touch_bonus != 0.0
    ):
        env = RewardShapingWrapper(
            env,
            reward_scale=reward_scale,
            living_reward=living_reward,
            dense_distance_coef=dense_distance_coef,
            touch_bonus=touch_bonus,
            touch_threshold=touch_threshold,
        )
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
