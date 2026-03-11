import torch as th
import numpy as np
from sb3_contrib import MaskablePPO
from gym_env import ScoutEnv  # 确保你的环境文件名正确

def get_action_name(game, action_idx, current_hand, table_cards):
    """
    增加了具体牌值和牌型预览的解析函数
    """
    # --- SHOW 动作 ---
    if action_idx < game.OFFSET_SCOUT:
        start, length = game._decode_show(action_idx - game.OFFSET_SHOW)
        played_cards = current_hand[start : start + length]
        return f"SHOW (起始位:{start}, 长度:{length}, 牌型:{played_cards})"
    
    # --- SCOUT 动作 ---
    elif action_idx < game.OFFSET_SCOUT_SHOW:
        side, ins, flip = game._decode_scout(action_idx - game.OFFSET_SCOUT)
        side_str = "左" if side == 0 else "右"
        flip_str = "翻转" if flip else "不翻转"
        
        # 提取被取走的牌 (假设 table_cards 是元组列表)
        raw_card = table_cards[0] if side == 0 else table_cards[-1]
        target_val = raw_card[1] if flip else raw_card[0] # 处理翻转逻辑
        
        preview_hand = list(current_hand)
        preview_hand.insert(ins, target_val)
        
        return f"SCOUT (取桌上{side_str}侧:{target_val}, 插入位置:{ins}, {flip_str}, 结果手牌:{preview_hand})"
    
    # --- SCOUT & SHOW 动作 ---
    else:
        return "SCOUT & SHOW (特殊动作)"

def verify():
    # 1. 加载环境和模型
    env = ScoutEnv()
    model = MaskablePPO.load("/data/feiyu/code/boardGameAI/scout/models/best_diff_model_170000.zip")
    
    obs, info = env.reset()
    game = env.unwrapped.game
    
    print("="*60)
    print("      SCOUT AI 验证脚本 (上帝视角 - 可见所有手牌)")
    print("="*60)

    done = False
    while not done:
        # 1. 打印当前全局状态
        print(f"\n" + "="*40)
        print(f"[阶段: {game.phase}] | 轮到: P{game.current_player}")
        print(f"当前得分情况: {game.round_scores}")
        
        for i in range(4):
            role = "AI (P0)" if i == 0 else f"NPC (P{i})"
            hand = game._get_active_hand(i)
            marker = " ★" if game.current_player == i else ""
            print(f"{role}{marker}: {hand}")
        
        print(f"当前桌面牌: {game.table_cards} (属于 P{game.table_owner})")
        print("-" * 40)

        # 2. 如果轮到 AI (P0)
        if game.current_player == 0:
            current_hand = game._get_active_hand(0)
            table_cards = game.table_cards
            
            # 生成模型预测 (省略部分重复的推理代码...)
            action_masks = env.unwrapped._gen_mask()
            with th.no_grad():
                # obs_tensor = {k: th.as_tensor(v).unsqueeze(0).to(model.device) for k, v in obs.items()}
                obs_tensor = th.as_tensor(obs).unsqueeze(0).to(model.device) 
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
                masked_probs = (probs * action_masks) / ( (probs * action_masks).sum() + 1e-8)

            # 打印 Top 建议
            top_indices = np.argsort(masked_probs)[-5:][::-1]
            print("AI 动作建议 (置信度):")
            for idx in top_indices:
                if masked_probs[idx] > 0:
                    print(f"  - {get_action_name(game, idx, current_hand, table_cards)}: {masked_probs[idx]:.2%}")
            
            action = top_indices[0]
            chosen_action_str = get_action_name(game, action, current_hand, table_cards)
            print(f"\n>> AI 决定执行: {chosen_action_str}")
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            # 3. 打印行动后的结果反馈
            print(f"\n[行动反馈]")
            print(f"本次获得即时奖励: {reward:.2f}")
            print(f"AI 更新后手牌: {game._get_active_hand(0)}")
            print(f"AI 更新后得分: {game.round_scores[0]}")
            
            input("\n按回车键继续观察 NPC 行动...")
            
        else:
            # NPC 运行逻辑
            print(f"等待 NPC (P{game.current_player}) 行动...")
            # 这里的 pass 是因为 env.step 会自动驱动 NPC 直到轮到 P0 或结束
            pass
        
    print("\n" + "="*60)
    print("游戏结束！")
    print(f"最终得分: {game.round_scores}")
    print("="*60)

if __name__ == "__main__":
    verify()