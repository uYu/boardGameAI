import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ScoutTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=256, nhead=8, num_layers=4, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # 拆分特征索引 (根据你之前的 93 层定义)
        # 0-33 是手牌、桌面、背板、连接性 (序列相关)
        # 34-92 是统计、对手、历史 (全局相关)
        self.seq_channels = 34 
        self.global_channels = 93 - 34
        
        # 序列处理流
        self.seq_projection = nn.Linear(self.seq_channels, d_model)
        self.cls_token = nn.Parameter(th.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(th.randn(1, 17, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全局处理流 (简单的 MLP)
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # 最终融合层
        self.fc = nn.Sequential(
            nn.Linear(d_model + 128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # observations: (Batch, 93, 16)
        # 1. 拆分特征
        seq_info = observations[:, :self.seq_channels, :].transpose(1, 2) # (B, 16, 34)
        # 全局信息取平均（因为它们在 16 个宽度上是平铺重复的）
        global_info = observations[:, self.seq_channels:, 0] # (B, 59)

        # 2. 处理序列 (Transformer)
        b = seq_info.shape[0]
        x = self.seq_projection(seq_info)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = th.cat((cls_tokens, x), dim=1) + self.pos_embedding
        
        # Padding Mask: 检查手牌第一层是否为0
        padding_mask = th.cat([
            th.zeros((b, 1), dtype=th.bool, device=x.device),
            (observations[:, 0, :] == 0)
        ], dim=1)
        
        latent_seq = self.transformer(x, src_key_padding_mask=padding_mask)[:, 0, :]

        # 3. 处理全局 (MLP)
        latent_global = self.global_mlp(global_info)

        # 4. 融合
        return self.fc(th.cat([latent_seq, latent_global], dim=-1))

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
        features_extractor_class=AlphaDouMlpExtractor,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(pi=[512, 256], vf=[512, 256]) # 这里的 pi 和 vf 会接在 extractor 之后
    )

    # 创建模型（需配合你的 ScoutEnv）
    # model = PPO(AlphaDouPolicy, env, policy_kwargs=policy_kwargs, verbose=1)