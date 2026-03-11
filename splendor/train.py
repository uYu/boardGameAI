import os
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

# 导入你的环境、提取器和回调
from gym_env import SplendorEnv
from callbacks import SplendorEvalCallback
from model import SplendorResNetExtractor, AlphaDouPolicy

def train():
    # 1. 创建训练环境和独立的评估环境
    # 建议评估环境独立出来，避免训练时的 Monitor 状态干扰评估
    train_env = SplendorEnv()
    train_env = Monitor(train_env)
    
    eval_env = SplendorEnv()
    eval_env = Monitor(eval_env)

    # 2. 定义模型架构
    # 使用我们之前写的 SplendorDictNet 作为特征提取器
    policy_kwargs = dict(
        features_extractor_class=SplendorResNetExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        # 提取器已经把特征处理得很好了，后面的 pi 和 vf 只需要简单的层即可
        net_arch=dict(pi=[512, 512], vf=[512, 512])
    )

    # 3. 初始化 MaskablePPO
    model = MaskablePPO(
        AlphaDouPolicy,
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,    # 结构复杂了，稍微调低学习率更稳定
        gamma=0.995,            # 略微降低 gamma，让 AI 关注更近的中期目标
        n_steps=4096,           # 增加采样长度，花砖一局步数多，长视野很重要
        batch_size=2048,        # 进一步增大 batch，让 ResNet 的 BatchNormalization 更稳定
        ent_coef=0.01,         # 增加熵系数，鼓励 AI 尝试不同的开局和扣牌策略
        tensorboard_log="./splendor_tensorboard/"
    )

    # 4. 配置评估回调
    eval_callback = SplendorEvalCallback(
        eval_env=eval_env, 
        eval_freq=10000,       # 缩短评估间隔，更早看到新架构的效果
        n_eval_episodes=50,    # 每次评估 50 局足够观察趋势
        verbose=1
    )

    # 5. 开始学习
    print("🚀 架构升级版训练启动：Dict Obs + Entity Embeddings")
    model.learn(
        total_timesteps=1000000, 
        callback=eval_callback,
        progress_bar=True      # SB3 提供的进度条很香
    )

    # 6. 保存最终成果
    model.save("./models/splendor_entity_model_final")
    print("✅ 训练完成！")

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./splendor_tensorboard", exist_ok=True)
    
    train()