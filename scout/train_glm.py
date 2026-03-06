import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- 第一步：高级特征提取器 ---
def extract_advanced_features(hand_raw):
    vals = [card for card in hand_raw]
    num_cards = len(vals)
    if num_cards == 0: return None

    # 1. 基础统计
    res = {
        "mean_val": np.mean(vals),
        "std_val": np.std(vals),
        "min_val": np.min(vals),
        "max_val": np.max(vals),
        "range": np.max(vals) - np.min(vals)
    }

    # 2. 原始位置特征 (前12张牌的数值)
    # 决策树能处理数值，不需要像线性模型那样做 One-hot
    for i in range(12):
        res[f"pos_{i}"] = vals[i] if i < num_cards else -1

    # 3. 组合潜力特征
    max_str, cur_str = 1, 1
    max_set, cur_set = 1, 1
    gaps_of_1 = 0 # 类似 [3, 5] 这种差值为2的间隙数量
    
    for i in range(num_cards - 1):
        if vals[i+1] == vals[i] + 1: cur_str += 1
        else:
            max_str = max(max_str, cur_str)
            cur_str = 1
        
        if vals[i+1] == vals[i]: cur_set += 1
        else:
            max_set = max(max_set, cur_set)
            cur_set = 1
            
        if abs(vals[i+1] - vals[i]) == 2:
            gaps_of_1 += 1

    res["max_straight"] = max(max_str, cur_str)
    res["max_set"] = max(max_set, cur_set)
    res["gaps_of_1"] = gaps_of_1
    res["avg_diff"] = np.mean(np.abs(np.diff(vals))) if num_cards > 1 else 0
    
    return res

# --- 第二步：加载数据 ---
def load_dataset(file_path):
    print(f"正在从 {file_path} 加载原始数据...")
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            features = extract_advanced_features(item["hand"])
            if features:
                features["target_score"] = item["score"]
                data_list.append(features)
    return pd.DataFrame(data_list)

# --- 第三步：训练与验证 ---
def train_and_eval():
    df = load_dataset("/data/feiyu/code/scout/data/scout_raw_data.jsonl")
    
    X = df.drop(columns=['target_score'])
    y = df['target_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM 参数设置
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }

    print("开始训练 LightGBM 回归模型...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # 验证模型
    y_pred = model.predict(X_test)
    print("\n" + "="*30)
    print(f"验证集 MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"验证集 R2 Score: {r2_score(y_test, y_pred):.4f}")
    print("="*30)

    # 保存模型
    joblib.dump(model, "scout_lgbm_model.pkl")
    
    # --- 第四步：特征重要性可视化 ---
    lgb.plot_importance(model, max_num_features=15, importance_type='gain')
    plt.title("Feature Importance (Gain)")
    plt.show()


def compare_hands(hand_a_raw, hand_b_raw, model_path="/data/feiyu/code/scout/scout_lgbm_model.pkl"):
    # 1. 加载模型
    model = joblib.load(model_path)
    
    # 2. 提取特征并转化为 DataFrame (LightGBM 需要列名)
    feat_a = extract_advanced_features(hand_a_raw)
    feat_b = extract_advanced_features(hand_b_raw)
    
    df_test = pd.DataFrame([feat_a, feat_b])
    
    # 3. 预测期望得分
    scores = model.predict(df_test)
    
    print("\n" + "="*40)
    print(f"{'手牌方向':<10} | {'预测期望得分':<15}")
    print("-" * 40)
    print(f"{'A面 (原始)':<10} | {scores[0]:>15.4f}")
    print(f"{'B面 (翻转)':<10} | {scores[1]:>15.4f}")
    print("-" * 40)
    
    better_side = "A" if scores[0] > scores[1] else "B"
    diff = abs(scores[0] - scores[1])
    print(f"结论: 建议选择 {better_side} 面 (分差: {diff:.4f})")
    print("="*40)

# --- 测试用例设计 ---

# 案例 1: 明显的对比 (A面很碎，B面有长顺子)
# 注意：二元组第一个是 A 面，第二个是 B 面
test_cases = [
    # --- 顺子潜力类 (Straight Potential) ---
    {"name": "长顺子 vs 散牌", "hand": [[1,10], [2,9], [3,8], [4,7], [5,6], [9,1], [10,2]]},
    {"name": "间断顺子 [1,2,4,5] vs 连贯小顺", "hand": [[1,9], [2,8], [4,7], [5,6], [10,1], [10,2]]},
    {"name": "大数值顺子 [8,9,10] vs 小数值顺子 [1,2,3]", "hand": [[8,1], [9,2], [10,3], [1,10], [2,9], [3,8]]},
    {"name": "末端挂钩 [1,2,3,9] vs [2,3,4,1]", "hand": [[1,5], [2,5], [3,5], [9,1], [10,2]]},

    # --- 对子/炸弹类 (Set/Pair Potential) ---
    {"name": "三条 7 vs 两对 (2,2 和 9,9)", "hand": [[7,1], [7,2], [7,3], [2,9], [2,8], [9,2], [9,1]]},
    {"name": "四条 10 (巨型炸弹) vs 杂乱高分牌", "hand": [[10,1], [10,2], [10,3], [10,4], [1,10], [2,9], [3,8]]},
    {"name": "对子夹杂 [5,5,6,5,5] vs 纯碎牌", "hand": [[5,1], [5,2], [6,9], [5,3], [5,4], [1,10]]},
    {"name": "低位对子 (1,1) vs 高位单张 (10)", "hand": [[1,10], [1,9], [10,1], [9,2], [8,3]]},

    # --- 紧凑度与间距类 (Gaps & Density) ---
    {"name": "极度紧凑 [4,5,6,7] vs 极大跨度 [1,10,1,10]", "hand": [[4,1], [5,2], [6,3], [7,4], [1,10], [10,1]]},
    {"name": "等差数列 [2,4,6,8] vs 随机序列", "hand": [[2,1], [4,2], [6,3], [8,4], [1,9], [3,7]]},
    {"name": "两端大牌 vs 中间大牌", "hand": [[10,1], [1,10], [2,9], [3,8], [10,2]]},

    # --- 极端分布类 (Extreme Distributions) ---
    {"name": "全小牌 [1,2,1,2,3] vs 全大牌 [9,10,9,8]", "hand": [[1,9], [2,10], [1,8], [2,7], [3,6], [9,1], [10,2]]},
    {"name": "镜像对称牌 [1,2,3,3,2,1] 两面逻辑", "hand": [[1,10], [2,9], [3,8], [3,8], [2,9], [1,10]]},
    {"name": "单张大断层 [1,10,1] vs [10,1,10]", "hand": [[1,10], [10,1], [1,10], [5,5], [5,5]]},

    # --- 选面博弈类 (Flip Decisions) ---
    {"name": "一面有顺无对 vs 一面有对无顺", "hand": [[1,5], [2,5], [3,5], [8,8], [9,8], [10,8]]},
    {"name": "牺牲张数换取组合 (一面11张碎牌 vs 翻转后10张带大对)", "hand": [[1,10], [2,10], [3,10], [4,1], [5,2], [6,3], [7,4], [8,5], [9,6], [10,7]]},
    {"name": "Scout 目标牌 (中间夹着一个 10 的 1,2,?,4,5)", "hand": [[1,9], [2,8], [10,1], [4,7], [5,6]]},
    {"name": "边缘清空优势 (两头都是小牌 1,2... vs 两头都是大牌 9,10...)", "hand": [[1,10], [2,9], [5,5], [6,6], [9,2], [10,1]]},
    {"name": "密集点数 [5,5,5] vs 顺子点数 [4,5,6]", "hand": [[5,1], [5,2], [5,3], [4,9], [5,8], [6,7]]},
    {"name": "绝对烂牌 A vs 绝对烂牌 B (看模型如何止损)", "hand": [[1,9], [3,7], [5,5], [8,2], [10,1], [2,10], [4,8], [6,6]]}
]

def run_20_tests(model_path):
    for i, case in enumerate(test_cases):
        hand_a = [item[0] for item in case["hand"]]
        hand_b = [item[1] for item in case["hand"]]
        
        compare_hands(hand_a, hand_b, model_path)
        print(f"案例 {i+1}: {case['name']}")
        
if __name__ == "__main__":
    # train_and_eval()
    run_20_tests("/data/feiyu/code/scout/scout_lgbm_model.pkl")