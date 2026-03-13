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
    def __init__(self, observation_space, d_model=512, nhead=4, num_layers=3, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # 输入维度: (Batch, Channels, 16)
        self.in_channels = observation_space.shape[0]
        self.seq_len = 16
        self.d_model = d_model

        # 1. 输入投影层：将每张牌的 90+ 维特征转为 Transformer 的向量
        self.input_projection = nn.Linear(self.in_channels, d_model)
        
        # 2. 位置编码：因为 Scout 位置固定，我们使用可学习的位置编码
        self.pos_embedding = nn.Parameter(th.randn(1, self.seq_len, d_model))
        
        # 3. Transformer Encoder 层
        # batch_first=True 方便处理 (Batch, Seq, Dim) 数据
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # observations 形状: (Batch, Channels, 16)
        
        # 步骤 A: 准备 Padding Mask
        # 假设第一行特征 (h_mat[0]) 在没有牌的位置是 0
        # src_key_padding_mask 要求: 有牌的地方为 False, 补零的地方为 True
        # 我们取所有通道的和，如果全为 0，说明这一列是空位
        padding_mask = (observations.sum(dim=1) == 0) # (Batch, 16)
        
        # 步骤 B: 维度转换 (Batch, Channels, 16) -> (Batch, 16, Channels)
        x = observations.transpose(1, 2) 
        
        # 步骤 C: 投影到 d_model 维度并加上位置编码
        x = self.input_projection(x) # (Batch, 16, d_model)
        x = x + self.pos_embedding
        
        # 步骤 D: 进入 Transformer 编码
        # 传入 padding_mask，让 Self-Attention 忽略掉没有牌的列
        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask) # (Batch, 16, d_model)
        
        # 步骤 E: 聚合全局信息
        # 我们不直接 Flatten，而是使用 Masked Global Average Pooling
        # 只对非 Padding 的位置取平均
        mask_expanded = (~padding_mask).unsqueeze(-1).float() # (Batch, 16, 1)
        sum_features = (encoded * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        global_features = sum_features / count # (Batch, d_model)
        
        return self.output_layer(global_features)

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