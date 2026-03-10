import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game import ScoutGame
import random
import numpy as np

# ==========================================
# 2. Gym 环境类 (ScoutEnv)
# ==========================================
class ScoutEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = ScoutGame()
        
        self.action_space = spaces.Discrete(380) 

        # AlphaDou 风格修改：
        # 我们构建一个 (53, 16) 的矩阵：
        # 16 是手牌最大长度 (宽度)
        # 53 是特征维度 (高度)，包含：
        # - 10层: 手牌 One-hot
        # - 10层: 桌面牌 One-hot
        # - 2层:  手牌连接性 (是否成顺/成对)
        # - 10层: 全场消耗牌统计 (平铺)
        # - 21层: 全局信息 (得分、筹码、对手张数等，平铺)
        self.observation_space = spaces.Dict({
            "obs_matrix": spaces.Box(low=0, high=11, shape=(53, 16), dtype=np.float32),
            "phase": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })


    def _get_obs(self, p=0):
        # 容错处理
        if not hasattr(self.game, 'hands'): self.game.reset()
        
        # 获取当前玩家手牌数值 (1-10)
        h_vals = self.game._get_active_hand(p)
        
        # ==========================================
        # 1. 空间特征 (Spatial Features) - 针对每一张牌的局部特征
        # ==========================================
        
        # Hand One-hot: (10, 16) - 每一列代表一个位置，每一行代表数字 1-10
        h_mat = np.zeros((10, 16), dtype=np.float32)
        for i, v in enumerate(h_vals[:16]):
            h_mat[v-1, i] = 1.0

        # Table One-hot: (10, 16) - 桌面牌对齐
        # AlphaDou 建议将桌面牌放在与手牌相同的观察维度
        t_mat = np.zeros((10, 16), dtype=np.float32)
        raw_table_cards = [c[0] for c in self.game.table_cards]
        for i, v in enumerate(raw_table_cards[:16]):
            t_mat[v-1, i] = 1.0

        # 手牌连接性特征 (Scout 核心：相邻牌是否成对或成顺)
        connectivity = np.zeros((2, 16), dtype=np.float32)
        for i in range(len(h_vals) - 1):
            if h_vals[i] == h_vals[i+1]:
                connectivity[0, i] = 1.0  # 对子倾向
            if abs(h_vals[i] - h_vals[i+1]) == 1:
                connectivity[1, i] = 1.0  # 顺子倾向

        # ==========================================
        # 2. 全局特征平铺 (Broadcasting Features) - AlphaDou 的精髓
        # 将全局数值复制 16 份，填充到矩阵行中，让卷积核在任何位置都能读到全局状态
        # ==========================================
        
        # 消耗牌统计 (Spent): (10, 16)
        # 每一行代表 1-10 消耗了多少张，整行值相同
        spent_mat = np.zeros((10, 16), dtype=np.float32)
        for val, count in self.game.card_counts_spent.items():
            if 1 <= val <= 10:
                spent_mat[val-1, :] = count / 5.0 # 归一化

        # 玩家状态信息 (Global Info): (21, 16)
        # 包含：剩余手牌数、得分、筹码、Owner、Phase 等
        # 这里将你原来的 21 维 global_info 平铺到 16 个宽度
        g_info_raw = []
        for i in range(4):
            idx = (p + i) % 4
            g_info_raw.extend([
                len(self.game.hands[idx])/12.0, 
                self.game.collected_cards[idx]/45.0, 
                self.game.vp_tokens[idx]/20.0, 
                1.0 if self.game.scout_show_tokens[idx] else 0.0
            ])
        
        # Table Owner & Phase
        owner_oh = np.zeros(4)
        if self.game.table_owner != -1:
            owner_oh[(self.game.table_owner - p + 4) % 4] = 1.0
        g_info_raw.extend(owner_oh.tolist())
        g_info_raw.append(1.0 if self.game.table_owner == p else 0.0)
        
        global_mat = np.tile(np.array(g_info_raw).reshape(-1, 1), (1, 16)) # (21, 16)

        # ==========================================
        # 3. 最终组合 (Feature Fusion)
        # ==========================================
        
        # 总行数：10(Hand) + 10(Table) + 2(Connect) + 10(Spent) + 21(Global) = 53 行
        # 最终形状 (53, 16)
        obs_matrix = np.vstack([
            h_mat,          # 0-9
            t_mat,          # 10-19
            connectivity,   # 20-21
            spent_mat,      # 22-31
            global_mat      # 32-52
        ])

        # 为了适配 Gymnasium 空间，我们也保留简单的 Phase One-hot
        phase_oh = np.zeros(3)
        phase_oh[0 if self.game.phase == "PLAY" else 1] = 1.0

        return {
            "obs_matrix": obs_matrix.astype(np.float32), # (53, 16)
            "phase": phase_oh.astype(np.float32)
        }

    def reset(self, seed=None, options=None):
        self.game.reset()
        self._run_opponents(verbose=False)
        return self._get_obs(0), {"action_mask": self._gen_mask()}

    def _gen_mask(self):
        mask = np.zeros(380, dtype=np.int8)
        for a in self.game.get_legal_actions(0): mask[a] = 1
        return mask

    def _calculate_hand_strength(self, hand_cards):
        """
        计算手牌的结构强度。
        优化目标：
        1. 识别已有的对子(Same-value)和顺子(Sequence)。
        2. 识别潜在的组合（Gap-1 的邻近牌，如手牌中的 2, 3）。
        3. 惩罚拆散长序列的行为。
        """
        if not hand_cards or len(hand_cards) < 2:
            return 0
        
        strength = 0.0
        n = len(hand_cards)
        # 基础分：手牌越少，离胜利越近
        strength += (12 - n) * 1.5
        visited = [False] * n  # 标记是否已计入成组分，避免重复计算

        # --- 第一阶段：计算已成组的强度 (对子/顺子) ---
        i = 0
        while i < n:
            if visited[i]:
                i += 1
                continue
                
            # 1. 检测相同数字 (对子/三张/炸弹)
            count_same = 1
            while i + count_same < n and hand_cards[i + count_same] == hand_cards[i]:
                count_same += 1
            
            # 2. 检测顺子 (正向或反向)
            count_seq = 1
            # 正向顺子: 1, 2, 3
            while i + count_seq < n and hand_cards[i + count_seq] == hand_cards[i + count_seq - 1] + 1:
                count_seq += 1
            
            # 反向顺子: 3, 2, 1 (Scout 规则中手牌物理顺序固定，反向顺子同样强大)
            count_rev = 1
            while i + count_rev < n and hand_cards[i + count_rev] == hand_cards[i + count_rev - 1] - 1:
                count_rev += 1
                
            # 取当前位置起始的最优组合
            max_group = max(count_same, count_seq, count_rev)
            
            if max_group >= 2:
                # 权重公式：长度的 1.5 次方，鼓励凑长手。
                # 2张: 2.8分, 3张: 5.2分, 4张: 8.0分, 5张: 11.1分
                strength += (max_group ** 1.5)
                # 标记这些牌已访问
                for j in range(i, i + max_group):
                    visited[j] = True
                i += max_group
            else:
                i += 1

        # --- 第二阶段：计算潜在关联 (解决 Case 2) ---
        # 检查那些没有成组，但彼此相邻且数值差为 1 的牌 (如 [2, 3, 9] 中的 2和3)
        for k in range(n - 1):
            if not visited[k] or not visited[k+1]:
                diff = abs(hand_cards[k] - hand_cards[k+1])
                if diff == 1:
                    # 虽然目前只是孤零零的两个相邻数字，但它们是顺子的种子
                    strength += 0.8 
                elif diff == 0:
                    # 这里理论上在第一阶段会被 visited 标记，作为冗余保护
                    strength += 1.0

        # --- 第三阶段：数值权重 (可选) ---
        # 在 Scout 中，小牌（1, 2）通常比大牌（9, 10）更难出掉或更容易被压制
        # 这里的 strength 可以根据剩余手牌的平均值微调，但建议先观察前两步效果

        return strength

    def step(self, action, verbose=False):
        mask = self._gen_mask()
        # 记录行动前的状态
        old_hand_len = len(self.game._get_active_hand(0))
        
        if mask[action] == 0:
            return self._get_obs(0), -1.0, True, False, {"action_mask": mask}

        # 执行动作
        self.game.step(action)        
        
        # 1. 基础引导奖励：鼓励出牌（每出一张牌给 0.1，Scout 拿牌则会是 -0.1）
        new_hand_len = len(self.game._get_active_hand(0))
        reward = (old_hand_len - new_hand_len) * 0.1

        # 2. 运行对手回合 (Self-play 或 随机)
        if not self.game.done:
            self._run_opponents(verbose=verbose)

        # 3. 终局结算：这是最关键的奖励
        if self.game.done:
            my_final = self.game.round_scores[0]
            opponents = [self.game.round_scores[i] for i in range(1, 4)]
            avg_others = sum(opponents) / 3.0
            
            # 注意这里用 +=，把终局大奖加在最后一步动作上
            reward += (my_final - avg_others) * 1.0
            
            if my_final > max(opponents):
                reward += 5.0
            elif my_final < min(opponents):
                reward -= 2.0 # 适当增加一点垫底惩罚

        return self._get_obs(0), reward, self.game.done, False, {"action_mask": self._gen_mask()}

    def _calculate_break_penalty(self, hand, start, length):
        """惩罚拆散原本相连的牌"""
        penalty = 0.0
        if length <= 0: return 0
        first = hand[start]
        last = hand[start + length - 1]

        # 检查左邻居
        if start > 0 and (hand[start-1] == first or abs(hand[start-1] - first) == 1):
            penalty -= 2.0
        # 检查右邻居
        if start + length < len(hand) and (hand[start+length] == last or abs(hand[start+length] - last) == 1):
            penalty -= 2.0
        return penalty

    def _calculate_scout_efficiency_penalty(self, hand, card_val, chosen_ins):
        """惩罚没有插在最优位置的 Scout 行为"""
        strengths = []
        for i in range(len(hand) + 1):
            tmp = hand[:i] + [card_val] + hand[i:]
            strengths.append(self._calculate_hand_strength(tmp))

        max_s = max(strengths)
        actual_s = strengths[chosen_ins] if chosen_ins < len(strengths) else strengths[-1]

        # 如果没选最好的位置，扣分
        return (actual_s + 1e-8) / (max_s + 1e-8) 

    def _run_opponents(self, verbose=False):
        safe = 0
        while not self.game.done and self.game.current_player != 0:
            cp = self.game.current_player
            legal = self.game.get_legal_actions(cp)

            if not legal: 
                break

            shows = [a for a in legal if a < self.game.OFFSET_SCOUT]
            act = random.choice(shows) if (shows and random.random() < 0.7) else random.choice(legal)

            self.game.step(act)

            if verbose:
                print(f"[NPC Player {cp}] 采取行动: {act}")

            safe += 1
            if safe > 50: 
                # 【核心修复】：如果死循环保护触发，必须强制结束游戏！
                # 否则会导致回合未轮到 AI，但环境却要求 AI 下达指令的错位问题。
                print("Warning: NPC loop stuck. Forcing game end.")
                self.game.done = True
                self.game._calculate_scores()
                break

    def render_state(self, title=""):
        state = self.game.get_state()
        print(f"--- {title} --- (Phase: {state['phase']})")
        print(f"Table: {state['table_cards']} | Owner: P{state['table_owner']}")
        for i in range(4):
            marker = ">>" if i == self.game.current_player else "  "
            print(f"{marker} P{i}: {str(state['hands'][i]).ljust(45)} | Score: {self.game.round_scores.get(i,0)}")
        print("-" * 50)

# ==========================================
# 3. 运行演示
# ==========================================
if __name__ == "__main__":
    env = ScoutEnv()
    obs, info = env.reset()
    env.render_state("初始状态 (轮到 P0)")

    for i in range(30):
        mask = info["action_mask"]
        legal = np.where(mask == 1)[0]
        if len(legal) == 0: break
        
        # 模拟模型选择一个合法的 Show 动作（如果有的话）
        shows = [a for a in legal if a < 212]
        action = int(np.random.choice(shows if shows else legal))
        
        obs, reward, done, truncated, info = env.step(action, verbose=True)
        
        if done:
            print("\n*** 游戏结束 ***")
            print(f"最终得分统计: {env.game.round_scores}")
            break