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
    def __init__(self, opponent_model):
        super().__init__()
        self.game = ScoutGame()
        
        self.action_space = spaces.Discrete(380) 
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(93, 16), # 63层特征，宽度16
            dtype=np.float32
        )
        # self.opponent_policy = opponent_policy # 外部传入的对手策略模型
        self.opponent_model = opponent_model  # 👈 新增：存储对手模型

    def _get_obs(self, p=0):
        # 矩阵宽度固定为 16 (最大手牌数)
        W = 16
        obs_layers = []

        # 获取基础数据
        h_vals = self.game._get_active_hand(p)
        h_back_vals = [c[1] for c in self.game.hands[p]]
        raw_table_cards = [c[0] for c in self.game.table_cards]

        # ==========================================
        # 1. 基础手牌 & 桌面 (20行)
        # ==========================================
        h_mat = np.zeros((10, W), dtype=np.float32)
        for i, v in enumerate(h_vals[:W]):
            h_mat[v-1, i] = 1.0
                
        t_mat = np.zeros((10, W), dtype=np.float32)
        for i, v in enumerate(raw_table_cards[:W]):
            t_mat[v-1, i] = 1.0
            
        obs_layers.extend([h_mat, t_mat])

        # ==========================================
        # 2. 背面手牌 (10行)
        # ==========================================
        h_back_mat = np.zeros((10, W), dtype=np.float32)
        for i, v in enumerate(h_back_vals[:W]):
            h_back_mat[v-1, i] = 1.0
            
        obs_layers.append(h_back_mat)

        # ==========================================
        # 3. 连接性特征 - 强度编码 (4行)
        # ==========================================
        conn_mat = np.zeros((4, W), dtype=np.float32)
        def get_conn_strength(val_list):
            strengths = np.zeros((2, len(val_list)))
            for i in range(len(val_list)):
                # 对子连击数
                count = 1
                if i > 0 and val_list[i] == val_list[i-1]: count += 1
                if i < len(val_list)-1 and val_list[i] == val_list[i+1]: count += 1
                strengths[0, i] = min(count * 0.3, 1.0)
                
                # 顺子连击数
                s_count = 1
                if i > 0 and abs(val_list[i] - val_list[i-1]) == 1: s_count += 1
                if i < len(val_list)-1 and abs(val_list[i] - val_list[i+1]) == 1: s_count += 1
                strengths[1, i] = min(s_count * 0.3, 1.0)
            return strengths

        if len(h_vals) > 0:
            s = get_conn_strength(h_vals)
            conn_mat[0, :len(h_vals)] = s[0] # 现成对子强度
            conn_mat[1, :len(h_vals)] = s[1] # 现成顺子强度
            
            # 背面牌的潜在连接强度 (帮助评估是否该翻转)
            s_back = get_conn_strength(h_back_vals)
            conn_mat[2, :len(h_back_vals)] = s_back[0]
            conn_mat[3, :len(h_back_vals)] = s_back[1]
            
        obs_layers.append(conn_mat)

        # ==========================================
        # 4. 手牌缝隙匹配度 (4行) - 全新核心特征
        # ==========================================
        gap_mat = np.zeros((4, W), dtype=np.float32)
        if len(raw_table_cards) > 0:
            left_t = raw_table_cards[0]
            right_t = raw_table_cards[-1]
            
            def calculate_gap_score(hand, target_val):
                scores = np.zeros(W)
                for i in range(len(hand) + 1):
                    if i >= W: break
                    l_val = hand[i-1] if i > 0 else -10
                    r_val = hand[i] if i < len(hand) else -10
                    score = 0.0
                    if target_val == l_val or target_val == r_val: score += 0.5 # 成对
                    if abs(target_val - l_val) == 1 or abs(target_val - r_val) == 1: score += 0.4 # 成顺
                    if abs(target_val - l_val) == 1 and abs(target_val - r_val) == 1: score += 0.5 # 完美顺子嵌入
                    scores[i] = min(score, 1.0)
                return scores

            gap_mat[0] = calculate_gap_score(h_vals, left_t)       # 正面缝隙 vs 桌面左端
            gap_mat[1] = calculate_gap_score(h_vals, right_t)      # 正面缝隙 vs 桌面右端
            gap_mat[2] = calculate_gap_score(h_back_vals, left_t)  # 背面缝隙 vs 桌面左端
            gap_mat[3] = calculate_gap_score(h_back_vals, right_t) # 背面缝隙 vs 桌面右端
            
        obs_layers.append(gap_mat)

        # ==========================================
        # 5. 全局统计与安全绝张 (11行)
        # ==========================================
        spent_vec = np.array([self.game.card_counts_spent.get(v, 0)/5.0 for v in range(1, 11)])
        spent_mat = np.tile(spent_vec.reshape(10, 1), (1, W))
        
        safe_row = np.zeros((1, W), dtype=np.float32)
        for i, v in enumerate(h_vals[:W]):
            if self.game.card_counts_spent.get(v, 0) >= 5: 
                safe_row[0, i] = 1.0
                
        obs_layers.extend([spent_mat, safe_row])

        # ==========================================
        # 6. 对手状态 (12行)
        # ==========================================
        opp_mat = np.zeros((12, W), dtype=np.float32)
        for idx in range(3):
            opp_idx = (p + idx + 1) % 4
            opp_info = [
                len(self.game.hands[opp_idx]) / 12.0,                  # 手牌数
                self.game.scores[opp_idx] / 30.0,                      # 得分
                1.0 if self.game.has_scout_token[opp_idx] else 0.0,    # Token
                (idx + 1) / 3.0                                        # 相对位置
            ]
            opp_mat[idx*4 : (idx+1)*4, :] = np.tile(np.array(opp_info).reshape(4, 1), (1, W))
            
        obs_layers.append(opp_mat)

        # ==========================================
        # 7. 语义压缩后的历史动作矩阵 (32行)
        # ==========================================
        # 记录最近 8 步，每步占用 4 行
        num_history_steps = 8
        rows_per_step = 4
        hist_mat = np.zeros((num_history_steps * rows_per_step, W), dtype=np.float32)
        
        # 确保 get_recent_actions 不会报错（如果游戏刚开始历史记录不足，就返回空列表）
        recent_actions = self.game.get_recent_actions(n=num_history_steps)

        for i, act in enumerate(recent_actions):
            base = i * rows_per_step
            
            # 第1层: 谁执行的 + 动作类型 (Show 为 1.0, Scout 为 0.5)
            rel_p = (act['player'] - p) % 4
            act_type = 1.0 if act['type'] == 'show' else 0.5
            hist_mat[base, :] = (rel_p / 3.0) * act_type
            
            # 第2层: 动作张数/强度
            if act['type'] == 'show':
                hist_mat[base + 1, :] = len(act['cards']) / 5.0
            else:
                hist_mat[base + 1, :] = 0.2 # Scout动作固定强度
                
            # 第3层: 牌值特征
            if act['type'] == 'show':
                hist_mat[base + 2, :] = max(act['cards']) / 10.0
            else:
                hist_mat[base + 2, :] = act.get('card_taken', 0) / 10.0
                
            # 第4层: 得分/惩罚标记 (Scout给被拿牌的人送了1分)
            if act['type'] == 'scout':
                hist_mat[base + 3, :] = 1.0
                
        obs_layers.append(hist_mat)

        # ==========================================
        # 最终拼接与 Padding 遮盖
        # ==========================================
        obs = np.vstack(obs_layers)
        
        # 保险断言：确保正好是 93 层，防止后续修改时不小心弄错维度引发 SB3 报错
        assert obs.shape == (93, 16), f"Obs shape error: expected (93, 16), got {obs.shape}"

        # Padding 处理：将手牌长度之外的区域全部置为 0
        # 这是为了配合 Transformer 的 src_key_padding_mask
        hand_len = len(h_vals)
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
        
        # 非法动作严惩
        if mask[action] == 0:
            return self._get_obs(0), -10.0, True, False, {"action_mask": mask}

        # --- 1. 记录动作前的状态 ---
        old_hand_len = len(self.game.hands[p])
        old_potential = self.game.calculate_hand_potential(p)
        is_scout = action >= self.game.OFFSET_SCOUT

        # --- 2. 执行动作 ---
        self.game.step(action) 
        
        # --- 3. 记录动作后的状态 ---
        new_hand_len = len(self.game.hands[p])
        new_potential = self.game.calculate_hand_potential(p)
        potential_diff = new_potential - old_potential
        
        reward = 0.0

        # --- 4. 密集奖励 (Dense Reward) 逻辑 ---
        if is_scout:
            # 【SCOUT 奖励】：侧重“眼光”
            # 拿牌后，潜力增加 1 是正常的（多了一张单牌）
            # 如果增加 > 1，说明这张牌插在缝隙里凑成了对子或顺子
            if potential_diff > 1.5: 
                # 举例：[3,5]变[3,4,5]，潜力从2变9，diff=7。奖励显著。
                reward += 1.0 + (potential_diff * 0.5)
                if verbose: print(f"-> 🎯 优质插入奖励! Diff: {potential_diff:.1f}")
            else:
                # 如果只是瞎插，或者拿了张毫无关联的牌，给微小惩罚，防止囤牌
                reward -= 0.5
        else:
            # 【SHOW 奖励】：侧重“进度”
            # 只要成功打出牌，就给进度奖。打出的牌越多（炸弹/大顺子），奖金越高。
            cards_played = old_hand_len - new_hand_len
            reward += cards_played * 1.5 
            
            # 特殊情况：如果为了打这一手牌，拆散了手里更大的潜力牌（比如为了打对2，拆了3连顺）
            # 这种“拆大补小”的行为通过 potential_diff 为负值来体现
            if potential_diff < -8: 
                reward -= 1.0 # 稍微惩罚一下极其不理智的拆牌
                if verbose: print(f"-> ⚠️ 警告：拆散了强力组合!")

        # --- 5. 运行对手回合 ---
        if not self.game.done:
            self._run_opponents(verbose=verbose)

        # --- 6. 终局奖励 (Sparse Reward) 逻辑 ---
        if self.game.done:
            my_final = self.game.round_scores[0]
            opponents = [self.game.round_scores[i] for i in range(1, 4)]
            avg_others = sum(opponents) / 3.0
            
            # A. 基础分差奖励（放大权值，让 AI 对分数敏感）
            reward += (my_final - avg_others) * 2.0
            
            # B. 胜负名次奖励
            if my_final > max(opponents):
                reward += 10.0  # 第一名巨奖
            elif my_final == max(self.game.round_scores.values()):
                 reward += 5.0  # 并列第一
            elif all(my_final < s for s in opponents):
                reward -= 5.0   # 垫底惩罚

        return self._get_obs(0), reward, self.game.done, False, {"action_mask": self._gen_mask()}

    def set_opponent_model(self, model):
        """用于 Callback 动态更新对手"""
        self.opponent_model = model


    def _run_opponents(self, verbose=False):
        safe = 0
        while not self.game.done and self.game.current_player != 0:
            cp = self.game.current_player
            # 注意：这里需要获取当前 cp 的 mask
            mask = self._gen_mask() 
            
            if self.opponent_model is not None:
                obs = self._get_obs(cp)
                # 使用对手模型预测，关闭 deterministic 增加博弈多样性
                action, _ = self.opponent_model.predict(obs, action_masks=mask, deterministic=False)
            else:
                # 随机逻辑作为保底
                legal = np.where(mask == 1)[0]
                shows = [a for a in legal if a < self.game.OFFSET_SCOUT]
                action = random.choice(shows) if (shows and random.random() < 0.7) else random.choice(legal)

            self.game.step(action)
            safe += 1
            if safe > 60: 
                self.game.done = True
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