import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_env import ScoutEnv
from network import ScoutFullFeatureExtractor, AlphaDouPolicy
from network_transformers import ScoutTransformerExtractor
from callbacks_self import ScoutWinRateCallback, SelfPlayUpdateCallback

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
    # --- 1. 加载初始模型 ---
    old_model_path = "/root/boardGameAI/scout/models/best_diff_model_670000.zip"
    pretrained_model = MaskablePPO.load(old_model_path)
    
    # --- 2. 初始化环境并注入初始对手 ---
    # 训练环境
    env = ScoutEnv(opponent_model=pretrained_model)
    env = ActionMasker(env, mask_fn)
    
    # 评估环境
    eval_env = ScoutEnv(opponent_model=pretrained_model)
    eval_env = ActionMasker(eval_env, mask_fn)

    # --- 3. 创建新模型 (可以继承旧模型的参数开始练) ---
    policy_kwargs = dict(
        features_extractor_class=ScoutTransformerExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(pi=[512], vf=[512])
    )

    model = MaskablePPO(
        AlphaDouPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=linear_schedule(2e-5),
        tensorboard_log="./scout_logs/"
    )
    
    # 【可选】让新模型从旧模型的权重开始，而不是随机开始
    model.set_parameters(pretrained_model.get_parameters())

    # --- 4. 组合 Callbacks ---
    # 每 1万步评估一次 (看看能不能打赢现有的对手)
    win_rate_cb = ScoutWinRateCallback(eval_env, eval_freq=10000)
    
    # 每 10万步更新一次对手 (把对手变强)
    update_cb = SelfPlayUpdateCallback(
        update_freq=100000, 
        save_path="./models/selfplay_history/",
        eval_env=eval_env
    )

    # 开始学习
    model.learn(total_timesteps=5000000, callback=[win_rate_cb, update_cb])

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()