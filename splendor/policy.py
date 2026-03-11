from game import SplendorGame, COLORS
import random

class HeuristicPolicy:
    @staticmethod
    def choose_action(game: SplendorGame, player_idx: int):
        p = game.players[player_idx]
        
        # 1. 检查能不能买牌 (优先看 T3 -> T2 -> T1)
        for level in [3, 2, 1]:
            for i, card in enumerate(game.board[level]):
                if card and p.can_afford(card):
                    return ("buy", level, i)
        
        # 2. 检查能不能买手中的保留牌
        for i, card in enumerate(p.reserved):
            if p.can_afford(card):
                return ("buy_reserved", i)

        # 3. 如果买不起，尝试拿缺少的宝石
        # 先找一张“努力一下就能买到”的牌（这里简化为 T1 的第一张有效牌）
        target_card = None
        for level in [1, 2, 3]:
            for card in game.board[level]:
                if card:
                    target_card = card
                    break
            if target_card: break
            
        if target_card:
            needed_colors = []
            for c in COLORS:
                # 计算还差多少宝石（减去折扣和已有宝石）
                shortfall = max(0, target_card.cost[c] - p.bonuses[c] - p.gems[c])
                if shortfall > 0 and game.bank[c] > 0:
                    needed_colors.append(c)
            
            if len(needed_colors) >= 3:
                return ("take", needed_colors[:3])
            elif len(needed_colors) > 0:
                # 补齐到3种颜色，或者拿2个同色
                available = [c for c in COLORS if game.bank[c] > 0]
                to_take = list(set(needed_colors + random.sample(available, min(len(available), 3))))[:3]
                return ("take", to_take)

        # 4. 最后保底：随机拿
        available = [c for c in COLORS if game.bank[c] > 0]
        if len(available) >= 3:
            return ("take", random.sample(available, 3))
        elif len(available) > 0:
            return ("take", [available[0]])
            
        return ("pass",) # 实在没招了（极少发生）



def run_simulation(max_turns=100):
    game = SplendorGame(num_players=2)
    
    for turn in range(max_turns):
        p_idx = game.current_idx
        p = game.players[p_idx]
        
        # 使用启发式策略
        action = HeuristicPolicy.choose_action(game, p_idx)
        
        # 执行动作
        action_type = action[0]
        if action_type == "buy":
            game.buy_card(action[1], action[2])
            print(f"回合 {turn}: 玩家 {p_idx} 购买了 L{action[1]} 卡牌 | 总分: {p.score}")
        elif action_type == "buy_reserved":
            game.buy_card(0, action[1], from_reserved=True)
            print(f"回合 {turn}: 玩家 {p_idx} 购买了保留牌 | 总分: {p.score}")
        elif action_type == "take":
            game.take_gems(action[1])
        
        # 胜利判定
        if p.score >= 15:
            print(f"*** 玩家 {p_idx} 获胜！总耗时 {turn} 回合 ***")
            break
            
        game.next_turn()

if __name__ == "__main__":
    run_simulation(500)
