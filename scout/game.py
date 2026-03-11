import itertools
import random
import numpy as np

# ==========================================
# 1. 游戏逻辑类 (ScoutGame)
# ==========================================
class ScoutGame:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.max_hand_size = 16  # Scout 标准每人11-12张
        
        # 动作空间索引计算
        self.NUM_SHOW = (self.max_hand_size * (self.max_hand_size + 1)) // 2
        self.NUM_SCOUT = 2 * (self.max_hand_size + 1) * 2
        
        self.OFFSET_SHOW = 0 
        self.OFFSET_SCOUT = self.OFFSET_SHOW + self.NUM_SHOW 
        self.OFFSET_SCOUT_SHOW = self.OFFSET_SCOUT + self.NUM_SCOUT 
        self.ACTION_PASS = self.OFFSET_SCOUT_SHOW + self.NUM_SCOUT 
        self.total_action_space = self.ACTION_PASS + 1
        
        # 生成双面牌堆 (1-10 的组合，不重复)
        self.full_deck = list(itertools.combinations(range(1, 11), 2))
        self.reset()

    def reset(self):
        deck = self.full_deck.copy()
        random.shuffle(deck)
        # 4人游戏，每人11张，剩下1张移除或不使用
        self.hands = {i: deck[i*11:(i+1)*11] for i in range(self.num_players)}
        self.phase = "PLAY" 
        self.setup_choices = {i: 0 for i in range(self.num_players)} 
        self.table_cards = [] 
        self.table_owner = -1
        self.current_player = 0
        self.done = False
        
        # 计分统计
        self.collected_cards = {i: 0 for i in range(self.num_players)} # 吃掉的牌 (1张1分)
        self.vp_tokens = {i: 0 for i in range(self.num_players)}       # Scout 给出的筹码
        self.scout_show_tokens = {i: True for i in range(self.num_players)}
        self.round_scores = {i: 0 for i in range(self.num_players)}
        self.card_counts_spent = {i: 0 for i in range(1, 11)}

        self.scores = {i: 0 for i in range(self.num_players)} 
        self.has_scout_token = {i: True for i in range(self.num_players)} 
        
        # 必须初始化动作历史列表，否则 get_action_history_vec 会报错
        self.action_history = [] 
        
        # 之前的 card_counts_spent 建议改为全局统计
        self.card_counts_spent = {v: 0 for v in range(1, 11)}

    def _get_active_hand(self, player):
        """获取当前玩家视角下的手牌数值列表"""
        hand = self.hands[player]
        is_flipped = self.setup_choices.get(player, 0)
        # 如果翻转，则数值取元组第二个，且序列反转
        return [c[1] if is_flipped else c[0] for c in (reversed(hand) if is_flipped else hand)]

    def _evaluate_combo(self, cards):
        """评估组合强度: (长度, 类型[1:同数, 0:顺子], 最大值)"""
        if not cards: return None
        n = len(cards)
        if n == 1: return (1, 1, cards[0])
        
        # 检查是否为同数集合 (Set)
        if len(set(cards)) == 1: return (n, 1, cards[0])
        
        # 检查是否为顺子 (Sequence)
        is_asc = all(cards[i] + 1 == cards[i+1] for i in range(n-1))
        is_desc = all(cards[i] - 1 == cards[i+1] for i in range(n-1))
        if is_asc or is_desc: return (n, 0, max(cards))
        
        return None

    def _beats(self, combo1, combo2):
        """比较 combo1 是否能大过 combo2"""
        if combo2 is None: return True
        if combo1 is None: return False
        # 规则：长度优先；长度相同时，Set 大于 Sequence；类型相同时，数值大者胜
        return combo1 > combo2

    # --- 编解码逻辑 (保持紧凑) ---
    def _encode_show(self, start, length):
        idx = 0
        for l in range(1, self.max_hand_size + 1):
            for s in range(self.max_hand_size - l + 1):
                if l == length and s == start: return idx
                idx += 1
        return 0

    def _decode_show(self, code):
        idx = 0
        for l in range(1, self.max_hand_size + 1):
            for s in range(self.max_hand_size - l + 1):
                if idx == code: return s, l
                idx += 1
        return 0, 1

    def _encode_scout(self, side, insert_idx, flip):
        # side: 0, 1 | flip: 0, 1 | insert_idx: 0 to 16
        # 确保 insert_idx 不会超过 16 (max_hand_size)
        safe_ins = min(insert_idx, self.max_hand_size)
        # 每一个 side 占据 (max_hand_size + 1) * 2 个坑位
        per_side = (self.max_hand_size + 1) * 2
        return side * per_side + safe_ins * 2 + flip

    def _decode_scout(self, code):
        per_side = (self.max_hand_size + 1) * 2
        side = code // per_side
        rem = code % per_side
        insert_idx = rem // 2
        flip = rem % 2
        return side, insert_idx, flip

    def get_recent_actions(self, n=3):
        """返回最近 n 步的动作列表，如果不足 n 步则用空动作填充"""
        hist = list(self.action_history)
        # 如果历史不够长，往前填充空的动作字典
        while len(hist) < n:
            hist.insert(0, {'player': -1, 'type': 'none', 'cards': [], 'pos': None})
        # 返回最近的 n 个
        return hist[-n:]

    def step(self, action):
        action_type = 0.5 if action >= self.OFFSET_SCOUT else 1.0
        # 强度简单定义：牌的数量
        power = len(self.table_cards) if action_type == 1.0 else 1
        p = self.current_player
        self.action_history.append({'type': action_type, 'power': power, 'player': p})

        if action == self.ACTION_PASS:
            self.phase = "PLAY"
            self.current_player = (self.current_player + 1) % self.num_players
            return

        # 处理 SCOUT & SHOW (拿牌)
        if action >= self.OFFSET_SCOUT_SHOW:
            side, ins, flip = self._decode_scout(action - self.OFFSET_SCOUT_SHOW)
            self._perform_scout(p, side, ins, flip)
            self.scout_show_tokens[p] = False
            self.phase = "WAITING_FOR_SHOW" 
            return 
            
        # 处理纯 SCOUT
        elif action >= self.OFFSET_SCOUT:
            side, ins, flip = self._decode_scout(action - self.OFFSET_SCOUT)
            self._perform_scout(p, side, ins, flip)
            self.phase = "PLAY"
            self.current_player = (self.current_player + 1) % self.num_players
            
        # 处理 SHOW
        else:
            start, length = self._decode_show(action - self.OFFSET_SHOW)
            self._perform_show(p, start, length)
            self.phase = "PLAY"
            self.current_player = (self.current_player + 1) % self.num_players

        self._check_end_condition()

    def _perform_scout(self, player, side, insert_idx, flip):
        if not self.table_cards: return

        # 1. 结算 VP：给当前桌面的主人加分
        if self.table_owner != -1 and self.table_owner != player:
            self.vp_tokens[self.table_owner] += 1
            self.scores[self.table_owner] += 1 # 同步更新特征矩阵用的分数
        
        # 2. 拿牌逻辑
        card_tuple = self.table_cards.pop(0 if side == 0 else -1)
        
        hand = self.hands[player]
        if flip: card_tuple = (card_tuple[1], card_tuple[0])
        
        is_flipped = self.setup_choices.get(player, 0)
        if is_flipped:
            # 物理存储映射：翻转状态下，insert_idx 0 是列表末尾
            actual_pos = len(hand) - insert_idx
            hand.insert(actual_pos, (card_tuple[1], card_tuple[0]))
        else:
            hand.insert(insert_idx, card_tuple)

        # 【核心修复】：只有桌子空了才重置主人。如果还有剩余牌，主人不变。
        if len(self.table_cards) == 0:
            self.table_owner = -1


    def get_legal_actions(self, player):
        if self.done: return []
        actions = []
        hand_vals = self._get_active_hand(player)
        n = len(hand_vals)
        table_vals = [c[0] for c in self.table_cards]
        table_combo = self._evaluate_combo(table_vals)

        # A. SHOW 逻辑 (0-135)
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                combo = self._evaluate_combo(hand_vals[start:start+length])
                if combo and self._beats(combo, table_combo):
                    actions.append(self.OFFSET_SHOW + self._encode_show(start, length))

        # B. SCOUT 逻辑
        if self.table_cards and self.table_owner != player and n < self.max_hand_size:
            sides = [0, 1] if len(self.table_cards) > 1 else [0]
            for side in sides:
                for ins in range(n + 1): # 新牌可以插在 n+1 个位置
                    for flip in [0, 1]:
                        code = self._encode_scout(side, ins, flip)
                        
                        # 纯 SCOUT (OFFSET_SCOUT 之后)
                        scout_act = self.OFFSET_SCOUT + code
                        if scout_act < self.OFFSET_SCOUT_SHOW:
                            actions.append(scout_act)
                        
                        # SCOUT & SHOW (OFFSET_SCOUT_SHOW 之后)
                        if self.scout_show_tokens[player]:
                            sc_show_act = self.OFFSET_SCOUT_SHOW + code
                            if sc_show_act < self.ACTION_PASS:
                                actions.append(sc_show_act)

        # C. 兜底逻辑：如果什么都做不了，或者发生了编码溢出
        if not actions:
            actions.append(self.ACTION_PASS)
            
        return actions

    def _perform_show(self, player, start, length):
        """执行出牌：结算分数并清空桌面"""
        if self.table_cards:
            # 增加得分 (collected_cards)
            self.collected_cards[player] += len(self.table_cards)
            for card in self.table_cards:
                self.card_counts_spent[card[0]] += 1

        # 【核心修复】：真正移除手牌，并处理 is_flipped 状态下的映射
        hand = self.hands[player]
        is_flipped = self.setup_choices.get(player, 0)

        # 视图索引映射到物理列表索引
        if is_flipped:
            phys_start = len(hand) - start - length
        else:
            phys_start = start

        played_cards = hand[phys_start : phys_start + length]

        # 如果手牌是翻转的，打到桌面上的牌必须反转物理顺序，同时翻转元组内外
        # 以保证 c[0] 始终是朝上的数值
        if is_flipped:
            played_cards.reverse()
            played_cards = [(c[1], c[0]) for c in played_cards]

        self.table_cards = played_cards
        del hand[phys_start : phys_start + length] # 真正从手牌中移除

        self.table_owner = player

    def _check_end_condition(self):
        # 1. 有人打光手牌 -> 结束
        for i in range(self.num_players):
            if len(self.hands[i]) == 0:
                self.done = True
                self._calculate_scores()
                return
        
        # 2. 轮到你时，桌上的牌还是你出的 -> 没人压得过，你赢了 -> 结束
        next_p = self.current_player
        if self.phase == "PLAY" and self.table_cards and self.table_owner == next_p:
            self.done = True
            self._calculate_scores()

    def _calculate_scores(self):
        """核心修复：计算最终得分 = 收集牌 + VP筹码 - 剩余手牌"""
        is_loop_end = False
        next_p = self.current_player
        # 如果是因为“转了一圈没人压过”而结束
        if self.table_cards and self.table_owner == next_p:
            is_loop_end = True

        for i in range(self.num_players):
            pos = self.collected_cards[i] + self.vp_tokens[i]
            neg = len(self.hands[i])
            
            # 特殊规则：如果是触发“没人压过”的 Owner，手牌不扣分
            if is_loop_end and i == self.table_owner:
                neg = 0
            # 同样，如果某人打光了手牌，他也不扣分（本身就是0）
            
            self.round_scores[i] = pos - neg

    def get_action_history_vec(self):
        """返回最近3手动作的编码，用于特征矩阵的最后6行"""
        vec = np.zeros(6, dtype=np.float32)
        # 取最近3个动作，每个动作占2位：[动作类型, 强度]
        # 类型编码建议：0:None, 0.5:Scout, 1.0:Show
        recent = self.action_history[-3:]
        for i, action_info in enumerate(recent):
            vec[i*2] = action_info['type']  # 0.5 或 1.0
            vec[i*2 + 1] = action_info['power'] / 10.0 # 归一化强度
        return vec
        
    def get_state(self):
        return {
            "hands": {i: self._get_active_hand(i) for i in range(self.num_players)},
            "table_cards": [c[0] for c in self.table_cards],
            "table_owner": self.table_owner,
            "round_scores": self.round_scores,
            "phase": self.phase
        }
        
    def calculate_hand_potential(self, player):
        """
        计算手牌结构的“潜力值”。
        逻辑：识别手牌中所有的 Set(同数) 和 Sequence(顺子)，
        价值 = sum(组合长度的平方)。
        这会引导 AI 倾向于保留长组合，而不是拆散它们。
        """
        hand = self._get_active_hand(player)
        if not hand: return 0
        
        potential = 0
        i = 0
        while i < len(hand):
            # 1. 尝试寻找从 i 开始的最长 Set
            set_len = 1
            for j in range(i + 1, len(hand)):
                if hand[j] == hand[i]: set_len += 1
                else: break
            
            # 2. 尝试寻找从 i 开始的最长 Sequence (升序或降序)
            seq_len_asc = 1
            for j in range(i + 1, len(hand)):
                if hand[j] == hand[j-1] + 1: seq_len_asc += 1
                else: break
            
            seq_len_desc = 1
            for j in range(i + 1, len(hand)):
                if hand[j] == hand[j-1] - 1: seq_len_desc += 1
                else: break
                
            best_local_len = max(set_len, seq_len_asc, seq_len_desc)
            
            # 评分公式：长度的平方（鼓励更长的组合）
            # 例如：[5,5,5] = 9分, 而 [5,2,5,5] = 1+4=5分
            if best_local_len > 1:
                potential += (best_local_len ** 2)
                i += best_local_len
            else:
                potential += 1 # 单张牌 1 分
                i += 1
        return potential