import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
# 假设你的 policy 和 game 模块已正确实现
from policy import HeuristicPolicy

# ================= 数据与常量 =================
COLORS = ['white', 'blue', 'green', 'red', 'black']
COLORS_ONLY = ['white', 'blue', 'green', 'red', 'black']
ALL_RESOURCES = ['white', 'blue', 'green', 'red', 'black', 'gold']
DIFF_3_COMBOS = list(combinations(COLORS, 3))  # 10 种组合

MAX_VAL = 7 
GEM_MATRIX_SHAPE = (MAX_VAL, 6) 
GEM_STR_TO_IDX = {color: i for i, color in enumerate(ALL_RESOURCES)}

# 矩阵维度定义：56 行 (特征) x 21 列 (实体)
NUM_FEATURES = 84
NUM_ENTITIES = 22
MAX_GEM_THRESHOLD = 7
MAX_BONUS_THRESHOLD = 6

def _to_threshold_array(resource_data, max_threshold, colors_list):
    """
    将数值转换为多层二值化（阈值）特征
    例如：3个红宝石 -> [1, 1, 1, 0, 0, 0, 0]
    """
    arr = np.zeros(len(colors_list) * max_threshold, dtype=np.float32)
    if isinstance(resource_data, dict):
        for i, color in enumerate(colors_list):
            count = min(int(resource_data.get(color, 0)), max_threshold)
            if count > 0:
                arr[i * max_threshold : i * max_threshold + count] = 1.0
    else:
        # 兼容列表输入
        for i, count in enumerate(resource_data):
            if i >= len(colors_list): break
            c = min(int(count), max_threshold)
            if c > 0:
                arr[i * max_threshold : i * max_threshold + c] = 1.0
    return arr
class SplendorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.heuristic_policy = HeuristicPolicy()

        # 【核心优化 1】将复杂的 Dict 替换为单一的 2D 矩阵 Box
        # 形状: (56特征, 21实体)，数值全部归一化到 [0, 1] 之间，极其利于神经网络收敛
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_FEATURES, NUM_ENTITIES), dtype=np.float32
        )
        self.action_space = spaces.Discrete(45)
    def _build_entity_vector(self, entity_type, obj, player=None):
        """
        核心优化：统一 84 维特征向量编码
        """
        vec = np.zeros(NUM_FEATURES, dtype=np.float32)
        if obj is None:
            return vec

        vec[77] = 1.0  # Is_Present (存在标识)

        # 1. 实体类型 One-hot (78-83)
        type_idx_map = {
            'bank': 78, 'player': 79, 'opponent': 80, 
            'noble': 81, 'board_card': 82, 'reserved_card': 83
        }
        if entity_type in type_idx_map:
            vec[type_idx_map[entity_type]] = 1.0

        # 2. 宝石/成本特征 (0-41)
        if entity_type in ['bank', 'player', 'opponent']:
            # 银行和玩家看当前持有宝石
            gems = obj.gems if hasattr(obj, 'gems') else obj
            vec[0:42] = _to_threshold_array(gems, MAX_GEM_THRESHOLD, ALL_RESOURCES)
        elif entity_type in ['board_card', 'reserved_card']:
            # 卡牌看购买成本
            vec[0:42] = _to_threshold_array(obj.cost, MAX_GEM_THRESHOLD, ALL_RESOURCES)

        # 3. 奖励(Bonus)/需求特征 (42-71)
        if entity_type in ['player', 'opponent']:
            # 玩家持有的 Bonus
            vec[42:72] = _to_threshold_array(obj.bonuses, MAX_BONUS_THRESHOLD, COLORS_ONLY)
        elif entity_type == 'noble':
            # 贵族需求的是 Bonus，不是宝石，放在这个区间对齐
            vec[42:72] = _to_threshold_array(obj.requirement, MAX_BONUS_THRESHOLD, COLORS_ONLY)
        elif entity_type in ['board_card', 'reserved_card']:
            # 卡牌提供的单一颜色奖励
            if obj.color in COLORS_ONLY:
                c_idx = COLORS_ONLY.index(obj.color)
                vec[42 + c_idx * MAX_BONUS_THRESHOLD] = 1.0

        # 4. 分数、等级、状态 (72-76)
        if hasattr(obj, 'score'):
            vec[72] = obj.score / 15.0
        elif hasattr(obj, 'points'):
            vec[72] = obj.points / 15.0

        if entity_type in ['board_card', 'reserved_card']:
            # Level One-hot (73-75)
            if 1 <= obj.level <= 3:
                vec[73 + obj.level - 1] = 1.0
            # 是否买得起 (76)
            if player and player.can_afford(obj):
                vec[76] = 1.0

        return vec

    def _get_obs(self):
        """
        按照 AlphaDou 逻辑组装 22 个实体，形成 (84, 22) 矩阵
        """
        p = self.game.players[self.game.current_idx]
        opp = self.game.players[1 - self.game.current_idx]
        entities = []

        # 0: 银行
        entities.append(self._build_entity_vector('bank', self.game.bank))

        # 1-2: 玩家与对手
        entities.append(self._build_entity_vector('player', p))
        entities.append(self._build_entity_vector('opponent', opp))

        # 3-5: 贵族 (3个)
        for i in range(3):
            noble = self.game.nobles[i] if i < len(self.game.nobles) else None
            entities.append(self._build_entity_vector('noble', noble))

        # 6-17: 场上卡牌 (12个)
        for l in [1, 2, 3]:
            for i in range(4):
                card = self.game.board[l][i] if i < len(self.game.board[l]) else None
                entities.append(self._build_entity_vector('board_card', card, p))

        # 18-20: 玩家保留卡 (3个)
        for i in range(3):
            card = p.reserved[i] if i < len(p.reserved) else None
            entities.append(self._build_entity_vector('reserved_card', card, p))

        # 21: 全局博弈状态实体 (这是 30% 到 60% 胜率的关键)
        global_vec = np.zeros(NUM_FEATURES, dtype=np.float32)
        global_vec[77] = 1.0  # 存在标识
        global_vec[78] = 1.0  # 借用 bank 类型位作为全局标识
        
        # 核心特征：分差 (让 AI 知道现在是落后需要博一把，还是领先需要稳)
        score_diff = (p.score - opp.score + 15) / 30.0  # 映射到 [0, 1]
        global_vec[72] = score_diff
        
        # 进度特征：当前轮次 (鼓励后期 AI 拿高分卡)
        global_vec[73] = min(self.game.turn_count / 50.0, 1.0)
        
        # 资源紧缺度：银行还剩多少金宝石
        global_vec[0] = self.game.bank.get('gold', 0) / 5.0
        
        entities.append(global_vec)

        # 最终形状：(22, 84) -> 转置为 (84, 22) 以适应卷积层
        obs_matrix = np.vstack(entities).T
        return obs_matrix.astype(np.float32)

    def step(self, action):
        p0 = self.game.players[0]
        opp = self.game.players[1]

        prev_stats = {
            'score': p0.score,
            'bonuses': p0.bonuses.copy(),
            'gems_total': sum(p0.gems.values())
        }

        action_tuple = self._decode_action(action)
        target_card = self._get_target_card(action_tuple)
        success = self._execute_action(action)

        terminated = self.game.is_game_over()
        truncated = False

        if not terminated:
            self.game.next_turn()
            opp_action = self.heuristic_policy.choose_action(self.game, 1)
            self._execute_heuristic_action(opp_action)
            terminated = self.game.is_game_over()
            if self.game.turn_count >= 100: truncated = True
            self.game.next_turn()

        reward = self._compute_reward({
            'action_type': action_tuple[0],
            'target_card': target_card,
            'prev_stats': prev_stats,
            'success': success
        })

        # 注意：Gymnasium 规范要求 info 中不要放 action_mask，建议外界直接调 env.action_masks()
        return self._get_obs(), reward, terminated, truncated, {"is_success": success}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from game import SplendorGame
        self.game = SplendorGame(num_players=2) 
        return self._get_obs(), {}

    # --- 辅助方法 ---
    def _get_target_card(self, action_tuple):
        atype = action_tuple[0]
        if atype in ["buy", "reserve"]:
            return self.game.board[action_tuple[1]][action_tuple[2]]
        if atype == "buy_reserved":
            idx = action_tuple[1]
            p = self.game.players[self.game.current_idx]
            return p.reserved[idx] if idx < len(p.reserved) else None
        return None

    def _decode_action(self, action_idx):
        if 0 <= action_idx <= 9: return ("take", DIFF_3_COMBOS[action_idx])
        if 10 <= action_idx <= 14: return ("take_two", COLORS[action_idx - 10])
        if 15 <= action_idx <= 26: 
            i = action_idx - 15
            return ("buy", (i // 4) + 1, i % 4)
        if 27 <= action_idx <= 29: return ("buy_reserved", action_idx - 27)
        if 30 <= action_idx <= 41:
            i = action_idx - 30
            return ("reserve", (i // 4) + 1, i % 4)
        if 42 <= action_idx <= 44: return ("reserve_blind", action_idx - 41)
        return ("pass",)

    def _execute_action(self, action_idx):
        # 实际调用游戏逻辑的代码
        t = self._decode_action(action_idx)
        if t[0] == "take": return self.game.take_gems(t[1], count=1)
        if t[0] == "take_two": return self.game.take_gems([t[1]], count=2)
        if t[0] == "buy": return self.game.buy_card(t[1], t[2])
        if t[0] == "buy_reserved": return self.game.buy_card(0, t[1], from_reserved=True)
        if t[0] == "reserve": return self.game.reserve_card(t[1], t[2])
        if t[0] == "reserve_blind": return self.game.reserve_card(t[1], -1, is_blind=True)
        return False

    def reset(self, seed=None, options=None):
        # 1. 处理 seed（Gymnasium 标准）
        super().reset(seed=seed)
        
        # 2. 重新初始化你的游戏逻辑
        from game import SplendorGame
        self.game = SplendorGame(num_players=2) 
        
        # 3. 获取初始观测
        obs = self._get_obs()
        
        # 4. 关键：必须返回 (obs, info) 格式！
        # 如果你这里只写了 return obs 或者漏写了，就会报你那个错
        return obs, {}

    def _execute_heuristic_action(self, action_tuple):
        atype = action_tuple[0]
        if atype == "buy": return self.game.buy_card(action_tuple[1], action_tuple[2])
        if atype == "buy_reserved": return self.game.buy_card(0, action_tuple[1], from_reserved=True)
        if atype == "take":
            colors = action_tuple[1]
            return self.game.take_gems(colors, count=2 if len(colors)==1 else 1)
        if atype == "reserve": return self.game.reserve_card(action_tuple[1], action_tuple[2])
        return False

    def _compute_reward(self, info):
        reward = 0.0
        p = self.game.players[0]
        opp = self.game.players[1]
        
        # 1. 基础得分奖励 (权重2.0)
        reward += (p.score - info['prev_stats']['score']) * 0.1
        
        # # 2. 资产奖励 (鼓励买牌而不是单纯囤宝石)
        # if info['action_type'] == 'buy' and info['success']:
        #     reward += 0.5 
        #     if info['target_card'] and info['target_card'].level > 1:
        #         reward += 0.3 * info['target_card'].level

        # # 3. 贵族接近奖励
        # for noble in self.game.nobles:
        #     old_dist = sum(max(0, r - info['prev_stats']['bonuses'].get(c, 0)) for c, r in noble.requirement.items())
        #     new_dist = sum(max(0, r - p.bonuses.get(c, 0)) for c, r in noble.requirement.items())
        #     if new_dist < old_dist: reward += 0.4

        # 4. 终局奖励
        if self.game.is_game_over():
            if p.score > opp.score: reward += 4.0
            elif p.score < opp.score: reward -= 4.0

        return reward

    def action_masks(self):
        mask = np.zeros(45, dtype=bool)
        p = self.game.players[self.game.current_idx]
        total = sum(p.gems.values())
        
        # 宝石逻辑
        if total < 10:
            for i, combo in enumerate(DIFF_3_COMBOS):
                if all(self.game.bank[c] > 0 for c in combo): mask[i] = True
            for i, c in enumerate(COLORS):
                if self.game.bank[c] >= 4: mask[10 + i] = True
        
        # 买牌/保留逻辑 (基于 game 对象的合法性检查)
        for i in range(12):
            l, idx = (i // 4) + 1, i % 4
            card = self.game.board[l][idx]
            if card and p.can_afford(card): mask[15 + i] = True
            if card and len(p.reserved) < 3: mask[30 + i] = True
        
        for i in range(len(p.reserved)):
            if p.can_afford(p.reserved[i]): mask[27 + i] = True
            
        for i in range(3):
            if len(p.reserved) < 3 and len(self.game.decks[i+1]) > 0: mask[42 + i] = True
            
        return mask