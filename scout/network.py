import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 1. 残差块定义
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

# ==========================================
# 2. Scout AlphaDou 风格特征提取器 (修复版)
# ==========================================
class ScoutAlphaDouStyleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)
        
        # --- A. 手牌流 (Hand Stream) ---
        # 结构：卷积 -> 残差 -> 池化 -> 卷积 -> 残差 -> 池化
        self.hand_stream = nn.Sequential(
            nn.Conv1d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock1D(64),
            ResidualBlock1D(64),
            ResidualBlock1D(64),
            nn.MaxPool1d(2),  # 序列长度 16 -> 8
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128),
            ResidualBlock1D(128),
            ResidualBlock1D(128),
            nn.MaxPool1d(2),  # 序列长度 8 -> 4
            nn.Flatten()
        )
        
        # --- B. 桌面流 (Table Stream) ---
        self.table_stream = nn.Sequential(
            nn.Conv1d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock1D(64),
            ResidualBlock1D(64),
            ResidualBlock1D(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128),
            ResidualBlock1D(128),
            ResidualBlock1D(128),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        # --- C. 全局特征 MLP (合并 Spent, hand_counts, global_info) ---
        # 输入维度: spent(10) + hand_counts(10) + global_info(21) = 41
        self.global_mlp = nn.Sequential(
            nn.Linear(41, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # --- E. 最终融合层 ---
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256 + 128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # 1. 处理序列数据 (Batch, 1, 16, 10) -> (Batch, 10, 16)
        h_in = observations["hand"].squeeze(1).permute(0, 2, 1)
        t_in = observations["table"].squeeze(1).permute(0, 2, 1)
        
        # 2. 处理全局/非序列数据
        # 确保 spent, hand_counts, global_info 形状都是 (Batch, Dim)
        g_in = th.cat([
            observations["spent"], 
            observations["hand_counts"], 
            observations["global_info"]
        ], dim=1)
        
        # 3. 提取各部分特征
        h_feat = self.hand_stream(h_in)
        t_feat = self.table_stream(t_in)
        g_feat = self.global_mlp(g_in)
        # 4. 特征拼接 (维度对齐的关键点)
        combined = th.cat([h_feat, t_feat, g_feat], dim=1)
        
        # 5. 输出最终 Embedding
        return self.fusion(combined)