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
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(93, 16), # 63层特征，宽度16
            dtype=np.float32
        )

    def _get_obs(self, p=0):
        # 矩阵宽度固定为 16 (最大手牌数)
        W = 16
        
        # --- 1. 基础手牌 & 桌面 (20行) ---
        h_vals = self.game._get_active_hand(p)
        h_mat = np.zeros((10, W), dtype=np.float32)
        for i, v in enumerate(h_vals[:W]):
            h_mat[v-1, i] = 1.0
                
        t_mat = np.zeros((10, W), dtype=np.float32)
        raw_table_cards = [c[0] for c in self.game.table_cards]
        for i, v in enumerate(raw_table_cards[:W]):
            t_mat[v-1, i] = 1.0

        # --- 2. 背面手牌 (10行) ---
        h_back_vals = [c[1] for c in self.game.hands[p]]
        h_back_mat = np.zeros((10, W), dtype=np.float32)
        for i, v in enumerate(h_back_vals[:W]):
            h_back_mat[v-1, i] = 1.0

        # --- 3. 连接性 (4行: 对子、顺子、潜在对、潜在顺) ---
        conn_mat = np.zeros((4, W), dtype=np.float32)
        for i in range(len(h_vals) - 1):
            if h_vals[i] == h_vals[i+1]: conn_mat[0, i] = 1.0 # 现成对子
            if abs(h_vals[i] - h_vals[i+1]) == 1: conn_mat[1, i] = 1.0 # 现成顺子
        # 潜在组合 (中间隔一张的情况，用于引导 Scout 拿牌)
        for i in range(len(h_vals) - 2):
            if h_vals[i] == h_vals[i+2]: conn_mat[2, i] = 1.0
            if abs(h_vals[i] - h_vals[i+2]) == 1: conn_mat[3, i] = 1.0

        # --- 4. 全局统计 (10行: 1-10 消耗程度) ---
        spent_vec = np.array([self.game.card_counts_spent.get(v, 0)/5.0 for v in range(1, 11)])
        spent_mat = np.tile(spent_vec.reshape(10, 1), (1, W))

        # --- 5. 安全/绝张 (1行) ---
        safe_vec = np.zeros(10)
        for v in range(1, 11):
            if self.game.card_counts_spent.get(v, 0) >= 5: safe_vec[v-1] = 1.0
        # 将绝张信息平铺到对应手牌位置上
        safe_row = np.zeros((1, W))
        for i, v in enumerate(h_vals[:W]):
            if safe_vec[v-1] == 1.0: safe_row[0, i] = 1.0

        # --- 6. 对手状态 (12行: 3个对手 x 4个特征) ---
        # 特征包含: 张数/12, 得分/30, 是否有Scout Token, 相对位置
        opp_mat = np.zeros((12, W), dtype=np.float32)
        for idx in range(3):
            opp_idx = (p + idx + 1) % 4
            opp_info = [
                len(self.game.hands[opp_idx]) / 12.0,
                self.game.scores[opp_idx] / 30.0,
                1.0 if self.game.has_scout_token[opp_idx] else 0.0,
                (idx + 1) / 3.0
            ]
            opp_mat[idx*4 : (idx+1)*4, :] = np.tile(np.array(opp_info).reshape(4, 1), (1, W))

        # --- 7. 优化后的历史动作矩阵 (共 36 行) ---
        # 记录前 3 步动作，每步占用 12 行
        num_history_steps = 3
        history_rows_per_step = 12
        hist_mat = np.zeros((num_history_steps * history_rows_per_step, W), dtype=np.float32)

        # 这里假设你在 ScoutGame 类中新增了一个方法 get_recent_actions(n=3)
        # 它返回一个列表，例如: 
        # [{'player': 1, 'type': 'show', 'cards': [4, 5, 5]}, 
        #  {'player': 2, 'type': 'scout', 'pos': 'left', 'card_taken': 3}, ...]
        recent_actions = self.game.get_recent_actions(n=num_history_steps)

        for step_idx, act_info in enumerate(recent_actions):
            base_row = step_idx * history_rows_per_step
            
            # 1. 记录是哪个玩家执行的动作 (相对当前玩家 p 的位置，归一化并平铺)
            rel_p = (act_info['player'] - p) % 4
            hist_mat[base_row + 0, :] = rel_p / 3.0  
            
            # 2. 如果是 Show 动作，将打出的牌型绘制在 10 行的矩阵里
            if act_info['type'] == 'show':
                for i, v in enumerate(act_info['cards'][:W]):
                    # v 的范围是 1-10，减 1 映射到索引 0-9
                    row_idx = base_row + 1 + (v - 1)
                    hist_mat[row_idx, i] = 1.0
                    
            # 3. 如果是 Scout 动作，记录是从哪边抽的 (左: 0.5, 右: 1.0)
            elif act_info['type'] == 'scout':
                scout_val = 0.5 if act_info['pos'] == 'left' else 1.0
                hist_mat[base_row + 11, :] = scout_val

        # --- 最终拼接 ---
        # 注意：因为去掉了之前的 6 行 hist_mat，加入了新的 36 行
        # 你的总行数现在应该是 57 (基础) + 36 = 93 行
        obs = np.vstack([
            h_mat, t_mat, h_back_mat, conn_mat, spent_mat, safe_row, opp_mat, hist_mat
        ])
        hand_len = len(self.game._get_active_hand(p))
        if hand_len < 16:
            obs[:, hand_len:] = 0.0
        return obs.astype(np.float32)


    def reset(self, seed=None, options=None):
        self.game.reset()
        self._run_opponents(verbose=False)
        return self._get_obs(0), {"action_mask": self._gen_mask()}

    def _gen_mask(self):
        # 初始化
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # 【新增的终局保护】如果游戏已经结束，无需打印报警，直接开放 PASS 动作占位即可
        if self.game.done:
            mask[self.game.ACTION_PASS] = True
            return mask
            
        # 动态获取当前玩家
        curr_p = self.game.current_player
        legal_actions = self.game.get_legal_actions(curr_p)
        
        # 填充合法动作
        if legal_actions:
            for a in legal_actions:
                mask[a] = True
                
        # 【核心修复】万一 legal_actions 为空，或者全被屏蔽了 (只有游戏进行中才需报警)
        if not np.any(mask):
            # 强制开启 PASS 动作，确保概率分布不为零
            mask[self.game.ACTION_PASS] = True
            
            # 记录罕见 Case 到日志，方便后续分析
            print(f"!!! [WARNING] All-False mask detected for Player {curr_p}!")
            print(f"Hand: {self.game.hands[curr_p]}, Table: {self.game.table_cards}")
            print(f"Table Owner: {self.game.table_owner}, Current Player: {curr_p}")
            print ("--------------------------------")
            
        return mask

    def step(self, action, verbose=False):
        mask = self._gen_mask()
        p = self.game.current_player
        # 记录行动前的状态
        old_potential = self.game.calculate_hand_potential(p)

        if mask[action] == 0:
            return self._get_obs(0), -1.0, True, False, {"action_mask": mask}

        # 执行动作
        self.game.step(action) 
        
        # 1. 基础引导奖励：鼓励出牌（每出一张牌给 0.1，Scout 拿牌则会是 -0.1）
        reward =  0. #(old_hand_len - new_hand_len) * 0.1
        new_potential = self.game.calculate_hand_potential(p)
        # 结构被优化（如连成了更长的顺子）给小奖，被拆散给惩罚
        potential_diff = (new_potential - old_potential) * 0.05 
        reward += potential_diff
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