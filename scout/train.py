import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_env import ScoutEnv
from network import ScoutAlphaDouStyleExtractor, AlphaDouPolicy
from callbacks import ScoutWinRateCallback

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性学习率衰减。
    :param initial_value: 初始学习率。
    :return: 一个函数，根据剩余进度计算当前学习率。
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remaining 从 1.0 (开始) 降到 0.0 (结束)
        """
        return progress_remaining * initial_value
    return func

def mask_fn(env):
    return env.unwrapped._gen_mask()

def train():
    # 环境初始化
    env = ScoutEnv()
    env = ActionMasker(env, mask_fn)

    eval_env = ScoutEnv()
    eval_env = ActionMasker(eval_env, mask_fn)

    # 策略参数：使用自定义提取器 + 深度 MLP
    policy_kwargs = dict(
        features_extractor_class=ScoutAlphaDouStyleExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(pi=[512, 256], vf=[512, 256]) # 这里的 pi 和 vf 会接在 extractor 之后
    )

    callback = ScoutWinRateCallback(eval_env, eval_freq=10000, n_eval_episodes=100, verbose=1)

    model = MaskablePPO(
        AlphaDouPolicy, # 因为输入是 Dict，必须用 MultiInputPolicy
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=1024,       # 卷积网络建议稍大的 batch
        learning_rate=linear_schedule(1e-4),   # 降低学习率保证卷积层稳定
        gamma=0.995,
        ent_coef=0.05,
        tensorboard_log="./scout_cnn_logs/"
    )

    print("开始基于 CNN 架构的强化学习训练...")
    model.learn(total_timesteps=5000000, callback=callback)

if __name__ == "__main__":
    train()