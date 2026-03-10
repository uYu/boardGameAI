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

class ScoutAlphaDouStyleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 1024):
        # 这里的 features_dim 必须和 policy_kwargs 里的匹配
        super().__init__(observation_space, features_dim)
        
        # 输入通道数是 53 (根据我们之前定义的矩阵行数)
        # 结构：卷积 -> 残差层 -> 压扁
        self.cnn_backbone = nn.Sequential(
            nn.Conv1d(53, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128),
            ResidualBlock1D(128),
            ResidualBlock1D(128),
            # 16 -> 8
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock1D(256),
            ResidualBlock1D(256),
            nn.Flatten() # 输出维度: 256 * 8 = 2048
        )
        
        # 处理辅助信息 phase (3维)
        self.phase_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        
        # 最终融合层: 2048 (CNN) + 64 (Phase) = 2112
        self.fusion = nn.Sequential(
            nn.Linear(2112, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # 1. 核心修复：现在从 observations 中取 "obs_matrix"
        # 形状: (Batch, 53, 16)
        x = observations["obs_matrix"]
        
        # 2. 提取特征
        cnn_out = self.cnn_backbone(x)
        phase_out = self.phase_mlp(observations["phase"])
        
        # 3. 拼接并输出
        combined = th.cat([cnn_out, phase_out], dim=1)
        return self.fusion(combined)

# ==========================================
# 3. AlphaDou 策略网络 (Custom Policy)
# 实现论文中的胜率 (WinRate) 与 期望 (Expectation) 的分离预测
# ==========================================
# ==========================================
# 1. 定义一个适配 SB3 的双输出 Identity 层
# ==========================================
# ==========================================
# 1. 完善后的双输出 Identity 层
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