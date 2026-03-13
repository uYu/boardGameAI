import json
import tqdm
import numpy as np
from gym_env import ScoutEnv
from sb3_contrib import MaskablePPO
import pandas as pd
from scipy.stats import pearsonr

def save_raw_samples(model_path, num_episodes=5000, output_file="/data/feiyu/code/boardGameAI/scout/data/scout_raw_data.jsonl"):
    """
    导出原始手牌信息。
    格式: JSON Lines (每行一个 JSON 对象，方便大文件读取)
    """
    env = ScoutEnv()
    model = MaskablePPO.load(model_path)
    
    print(f"开始采集原始数据，目标局数: {num_episodes}")

    with open(output_file, "w", encoding="utf-8") as f:
        for _ in tqdm.tqdm(range(num_episodes)):
            obs, info = env.reset()
            game = env.unwrapped.game
            
            # 1. 记录初始时刻四个人的原始手牌
            # 假设 game.hands[i] 结构为 [(up1, down1), (up2, down2), ...]
            initial_hands = []
            for i in range(4):
                # 深度拷贝一份原始手牌，防止后续对局修改了对象
                hand_data = [list(card) for card in game.hands[i]]
                initial_hands.append(hand_data)
            
            # 2. 模拟对局直到结束
            done = False
            while not done:
                action_masks = env.unwrapped._gen_mask()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # 3. 记录结算得分
            final_scores = game.round_scores
            
            # 4. 封装并写入数据
            # 一局游戏产生一个 entry，包含 4 个玩家的独立视角
            entries = []
            for i in range(4):
                entry = {
                    "hand": [item[0] for item in initial_hands[i]],   # 原始手牌 [[7,1], [8,2], ...]
                    "score": int(final_scores[i]), # 最终得分
                    "player_id": i
                }
                entries.append(entry)
            f.write(json.dumps(entries) + "\n")

    print(f"\n采集完毕！原始数据已保存至: {output_file}")

def extract_manual_features(hand_raw):
    """
    输入: [[val_up, val_down], ...]
    输出: 一个包含各种人工特征的字典
    """
    # 提取当前面（第一个元素）
    vals = [card for card in hand_raw]
    num_cards = len(vals)
    
    if num_cards == 0:
        return None

    # 1. 基础特征
    mean_val = np.mean(vals)
    max_val = np.max(vals)
    min_val = np.min(vals)
    
    # 2. 统计特征 (1-10的数量)
    counts = np.zeros(10)
    for v in vals:
        counts[min(max(int(v), 1), 10) - 1] += 1
    high_card_count = np.sum(counts[7:]) # 8, 9, 10 的张数

    # 3. 连张与对子检测
    max_straight = 1
    current_straight = 1
    max_set = 1
    current_set = 1
    
    for i in range(len(vals) - 1):
        # 连张 (1, 2, 3...)
        if vals[i+1] == vals[i] + 1:
            current_straight += 1
        else:
            max_straight = max(max_straight, current_straight)
            current_straight = 1
            
        # 对子 (7, 7, 7...)
        if vals[i+1] == vals[i]:
            current_set += 1
        else:
            max_set = max(max_set, current_set)
            current_set = 1
            
    max_straight = max(max_straight, current_straight)
    max_set = max(max_set, current_set)

    # 4. 紧凑度 (相邻牌差值的平均值)
    avg_diff = np.mean(np.abs(np.diff(vals))) if num_cards > 1 else 0

    return {
        "num_cards": num_cards,        # 总张数
        "mean_val": mean_val,          # 平均点数
        "max_val": max_val,            # 最大单张
        "high_cards": high_card_count, # 大牌数量
        "max_straight": max_straight,  # 最长连张长度
        "max_set": max_set,            # 最大对子长度
        "avg_diff": avg_diff           # 序列紧凑度
    }

def analyze_correlation(file_path):
    features_list = []
    scores = []

    print(f"正在读取并处理原始数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            feat = extract_manual_features(data["hand"])
            if feat:
                features_list.append(feat)
                scores.append(data["score"])

    df = pd.DataFrame(features_list)
    df['final_score'] = scores

    print("\n" + "="*40)
    print("      特征与最终得分的相关系数分析")
    print("="*40)
    print(f"{'特征名称':<15} | {'相关系数 (r)':<15} | {'强度'}")
    print("-" * 45)

    correlations = {}
    for col in df.columns:
        if col == 'final_score':
            continue
        
        # 计算皮尔逊相关系数
        r_val, p_val = pearsonr(df[col], df['final_score'])
        correlations[col] = r_val
        
        # 判断强度
        strength = "强" if abs(r_val) > 0.5 else "中" if abs(r_val) > 0.3 else "弱"
        print(f"{col:<15} | {r_val:>15.4f} | {strength}")

    return df, correlations


if __name__ == "__main__":
    MODEL_PATH = "/data/feiyu/code/boardGameAI/scout/models/best_diff_model_280000"
    save_raw_samples(MODEL_PATH)
    df_result, corr_result = analyze_correlation("/data/feiyu/code/scout/data/scout_raw_data.jsonl")
