import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
import json
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# 原始数据预处理 (假设 raw_data 已定义)
def create_pairs(raw_data):
    pairs = []
    for round_info in raw_data:
        # 对每一局中的 4 个玩家进行两两组合
        for p1, p2 in itertools.combinations(round_info, 2):
            hand1 = [v - 1 for v in p1['hand']]
            hand2 = [v - 1 for v in p2['hand']]
            if p1['score'] == p2['score']:
                continue # 跳过，不产生样本
            # 标签：如果 p1 分数高为 1，p2 高为 0 (忽略平局或按 0.5 处理)
            label = 1.0 if p1['score'] > p2['score'] else 0.0
            # if p1['score'] - p2['score'] > 10.:
            #     label = 1.0
            # if p1['score'] < p2['score'] + 10.:
            #     label = 0.
            pairs.append((hand1, hand2, label))
            pairs.append((hand2, hand1, 1 - label))

    return pairs

def prepare_pure_data(raw_data):
    pairs = []
    for round_info in raw_data:
        sorted_p = sorted(round_info, key=lambda x: x['score'], reverse=True)
        best_hand = torch.tensor([v for v in sorted_p[0]['hand']]) # 1-10直接用
        worst_hand = torch.tensor([v for v in sorted_p[-1]['hand']])
        
        # 正向：Best vs Worst -> 1
        pairs.append((best_hand, worst_hand, 1.0))
        pairs.append((worst_hand, best_hand, 0.0))
    return pairs

class ScoutPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h1, h2, label = self.pairs[idx]
        return torch.tensor(h1), torch.tensor(h2), torch.tensor([label], dtype=torch.float32)
# --- 日志配置 ---


class ScoutPureRanker(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=16):
        super(ScoutPureRanker, self).__init__()
        
        # 1. 词嵌入
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim) # 0留给padding
        
        # 2. 多尺度卷积层：模拟人类看牌
        # kernel=2 扫描对子/邻次；kernel=3 扫描顺子/三连
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=2)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=3)
        
        self.relu = nn.ReLU()
        
        # 3. 决策全连接层
        # 输入是 (32+32) = 64 维特征
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _get_features(self, x):
        # x: [batch, 11]
        x = self.embedding(x).permute(0, 2, 1) # [batch, embed_dim, 11]
        # 提取两种尺度的特征并进行全局池化
        feat2 = self.relu(self.conv2(x))
        feat2 = torch.max(feat2, dim=2)[0] # Global Max Pooling
        
        feat3 = self.relu(self.conv3(x))
        feat3 = torch.max(feat3, dim=2)[0] # Global Max Pooling
        
        return torch.cat([feat2, feat3], dim=1) # [batch, 64]

    def forward(self, hand_a, hand_b):
        f_a = self._get_features(hand_a)
        f_b = self._get_features(hand_b)
        # 计算特征差异
        diff = f_a - f_b
        return self.classifier(diff)

# --- 数据预处理 (第一名 vs 最后一名 + 对称增强) ---


def train():
    with open('/data/feiyu/code/boardGameAI/scout/data/scout_raw_data.jsonl') as f:
        raw_data = [json.loads(line) for line in f][: 4000]
        
    pairs_data = prepare_pure_data(raw_data)
    dataset = ScoutPairDataset(pairs_data)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    model = ScoutPureRanker()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(20):
        total_loss = 0
        for h_a, h_b, label in dataloader:
            optimizer.zero_grad()
            # 预测 A 胜过 B 的概率
            output = model(h_a, h_b)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    model_path = "/data/feiyu/code/scout/data/flip_model.path"
    torch.save(model.state_dict(), model_path)

def dev():
    # --- 1. 配置 ---
    model_path = '/data/feiyu/code/scout/data/flip_model.path'  # 预训练模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. 准备测试数据 (这里用你提供的数据作为示例) ---
    # 实际开发时，这里应该是你保留的验证集/测试集
    with open('/data/feiyu/code/boardGameAI/scout/data/scout_raw_data.jsonl') as f:
        raw_test_data = [json.loads(line) for line in f][4000: ]
    pairs_data = prepare_pure_data(raw_test_data)
    dataset = ScoutPairDataset(pairs_data)
    # test_samples = []
    # for round_info in raw_test_data:
    #     # 假设 A 是第一名，B 是最后一名
    #     h_a = torch.tensor(round_info[0]['hand'])
    #     h_b = torch.tensor(round_info[1]['hand'])
    #     label = 1.0 if round_info[0]['score'] > round_info[1]['score'] else 0.0
    #     test_samples.append((h_a, h_b, label))

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- 3. 加载模型结构 ---
    # 注意：ScoutPureRanker 的类定义必须在作用域内可用
    model = ScoutPureRanker().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print(f"📦 成功加载模型: {model_path}")
    else:
        print(f"❌ 未找到模型文件: {model_path}")
        return

    # --- 4. 执行预测与 Acc 计算 ---
    correct = 0
    total = len(test_loader)
    print ('===', total)
    
    print(f"🔍 开始验证流程 (Total: {total})...")
    print("-" * 30)

    with torch.no_grad():
        for i, (h_a, h_b, label) in enumerate(test_loader):
            h_a, h_b = h_a.to(device), h_b.to(device)
            
            # 模型预测 A 比 B 好的概率
            prob = model(h_a, h_b).item()
            
            # 二分类判断
            pred = 1.0 if prob > 0.5 else 0.0
            is_right = (pred == label.item())
            
            if is_right:
                correct += 1
            
            # 打印每一条的预测细节
            status = "✅" if is_right else "❌"
            # print(f"Sample {i+1:02d} | Prob: {prob:.4f} | Pred: {int(pred)} | Target: {int(label.item())} | {status}")

    # --- 5. 最终统计 ---
    acc = correct / total if total > 0 else 0
    print("-" * 30)
    print(f"📊 Final Test Accuracy: {acc:.2%}")
    
    return acc



if __name__ == "__main__":
    train()
    dev()