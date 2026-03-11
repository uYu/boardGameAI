import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# ==========================================
# 1. 基础组件：一维残差块 (与 Scout 保持一致)
# ==========================================
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SplendorResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=512):
        # 现在的 observation_space 应该是 (84, 22)
        super().__init__(observation_space, features_dim)

        in_channels = 84  # 特征数
        seq_len = 22      # 实体数

        # ---------------------------------------------------------
        # 第一阶段：实体内部特征提取 (Pointwise Convolution)
        # 作用：让网络先理解单个实体的 84 维编码含义，降低后续计算复杂度
        # ---------------------------------------------------------
        self.input_stem = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=1), # 1x1 卷积融合特征
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # ---------------------------------------------------------
        # 第二阶段：实体间关系学习 (ResNet Layers)
        # 作用：让“玩家”列与“卡牌”列进行交互，判断“够不够买”
        # ---------------------------------------------------------
        self.res_layers = nn.Sequential(
            ResidualBlock1D(128),
            ResidualBlock1D(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock1D(256),
            ResidualBlock1D(256)
        )

        # ---------------------------------------------------------
        # 第三阶段：展平与全局映射
        # 展平维度：256 (channels) * 22 (entities) = 5632
        # ---------------------------------------------------------
        flatten_dim = 256 * seq_len

        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 1024),
            nn.LayerNorm(1024), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, features_dim),
            nn.LayerNorm(features_dim), # 再次标准化，防止 PPO 训练中后期梯度爆炸
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: [Batch, 84, 22]
        x = self.input_stem(observations)
        x = self.res_layers(x)
        return self.output_layer(x)

# 残差块保持不变，但建议增加一个缩放系数（AlphaDou 常用技巧）
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
        return self.relu(x + self.conv(x))


# ==========================================
# 3. AlphaDou 策略网络 (Policy & Value 分离)
# ==========================================
class AlphaDouMlpExtractor(nn.Module):
    def __init__(self, features_dim: int):
        super().__init__()
        # SB3 必须要求的维度属性
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return features


class AlphaDouPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(AlphaDouPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """覆盖 SB3 默认的 MLP 提取器，使用免冗余的 AlphaDou 提取器"""
        self.mlp_extractor = AlphaDouMlpExtractor(self.features_dim)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # 胜率预测头 (可选，如果自定义了 loss 就可以用到)
        self.win_rate_head = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

# ==========================================
# 4. 训练启动示例
# ==========================================
if __name__ == "__main__":
    from sb3_contrib import MaskablePPO
    # 假设你的环境导入如下：
    # from env import SplendorEnv

    # env = SplendorEnv()

    # 策略参数配置
    policy_kwargs = dict(
        features_extractor_class=SplendorResNetExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        # pi (Actor) 和 vf (Critic) 分别接两层全连接
        net_arch=dict(pi=[512, 256], vf=[512, 256]) 
    )

    # 创建模型 (注意使用 MaskablePPO 支持动作掩码)
    # model = MaskablePPO(
    #     AlphaDouPolicy, 
    #     env, 
    #     policy_kwargs=policy_kwargs, 
    #     verbose=1,
    #     learning_rate=3e-4,
    #     batch_size=512
    # )

    # model.learn(total_timesteps=1000000)
    print("模型结构定义完成，随时可以开始训练！")