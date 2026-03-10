import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# ==========================================
# 1. 残差块定义 (Residual Block)
# ==========================================
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        return self.relu(out)

class ScoutFullFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256): # features_dim 从 512/1024 降到 256
        super().__init__(observation_space, features_dim)
        
        # 这里的思路是：保留 ResNet 结构，但把每一层的宽度切一半甚至更多
        self.cnn = nn.Sequential(
            # 输入 63 通道 -> 降到 64 通道 (之前是 128/256)
            nn.Conv1d(63, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 简化版残差组 1
            ResidualBlock1D(64),
            # 压缩长度 16 -> 8，通道数只增加到 128
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # 简化版残差组 2
            ResidualBlock1D(128),
            nn.Flatten() # 128 * 8 = 1024
        )
        
        # 最终映射层也变小
        self.output_layer = nn.Sequential(
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.output_layer(self.cnn(observations))

# ==========================================
# 3. AlphaDou 策略网络 (Custom Policy)
# 实现论文中的胜率 (WinRate) 与 期望 (Expectation) 的分离预测
# ==========================================
class AlphaDouMlpExtractor(nn.Module):
    def __init__(self, features_dim: int):
        super().__init__()
        # SB3 必须要求的维度属性
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features: th.Tensor):
        """同时返回 Policy 和 Value 的特征"""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """核心修复：SB3 内部会通过此方法获取策略特征"""
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """核心修复：SB3 内部会通过此方法获取价值特征"""
        return features

# ==========================================
# 2. 修改后的 AlphaDouPolicy
# ==========================================
class AlphaDouPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(AlphaDouPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        覆盖 SB3 默认的 MLP 提取器。
        使用我们自定义的 AlphaDouMlpExtractor。
        """
        self.mlp_extractor = AlphaDouMlpExtractor(self.features_dim)

    def _build(self, lr_schedule) -> None:
        # 调用父类构建标准结构 (action_net, value_net 等)
        super()._build(lr_schedule)
        
        # 可选：如果你想完全还原 AlphaDou，可以额外添加胜率头
        # 但在标准的 PPO.learn() 中，该头不会被自动更新，除非自定义训练逻辑
        self.win_rate_head = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
# ==========================================
# 4. 使用示例
# ==========================================
if __name__ == "__main__":
    from stable_baselines3 import PPO

    # 定义策略参数
    policy_kwargs = dict(
        features_extractor_class=ScoutAlphaDouExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(pi=[512, 256], vf=[512, 256]) # 这里的 pi 和 vf 会接在 extractor 之后
    )

    # 创建模型（需配合你的 ScoutEnv）
    # model = PPO(AlphaDouPolicy, env, policy_kwargs=policy_kwargs, verbose=1)