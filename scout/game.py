import itertools
import random

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
        stride = (self.max_hand_size + 1) * 2 
        return side * stride + insert_idx * 2 + flip

    def _decode_scout(self, code):
        stride = (self.max_hand_size + 1) * 2
        side = code // stride
        rem = code % stride
        return side, rem // 2, rem % 2
        
    def step(self, action):
        p = self.current_player

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
        """执行侦察：拿牌，给桌主分"""
        if not self.table_cards: return

        # 1. 给桌主加 1 VP (筹码分)
        if self.table_owner != -1:
            self.vp_tokens[self.table_owner] += 1
        
        # 2. 移除牌并加入手牌
        card_tuple = self.table_cards.pop(0 if side == 0 else -1)
        
        # --- 关键修复：不要在这里把 table_owner 设为 -1 ---
        # 即使 table_cards 空了，也要保留 table_owner，
        # 直到接下来的 SHOW 动作执行时，判定谁该拿走这些牌（或判定没人拿）。
        
        hand = self.hands[player]
        if flip: card_tuple = (card_tuple[1], card_tuple[0])
        is_flipped = self.setup_choices.get(player, 0)
        if is_flipped:
            hand.insert(len(hand) - insert_idx, (card_tuple[1], card_tuple[0]))
        else:
            hand.insert(insert_idx, card_tuple)
        if len(self.table_cards) == 0:
            self.table_owner = -1  # 桌面空了，不再属于任何人

    def get_legal_actions(self, player):
        if self.done: return []
        actions = []
        hand_vals = self._get_active_hand(player)
        n = len(hand_vals)

        table_vals = [c[0] for c in self.table_cards]
        table_combo = self._evaluate_combo(table_vals)

        # 情况 A: 处于 SCOUT & SHOW 的后续出牌阶段
        if self.phase == "WAITING_FOR_SHOW":
            for length in range(1, n + 1):
                for start in range(n - length + 1):
                    combo = self._evaluate_combo(hand_vals[start:start+length])
                    if combo and self._beats(combo, table_combo):
                        actions.append(self.OFFSET_SHOW + self._encode_show(start, length))
            if not actions: actions.append(self.ACTION_PASS)
            return actions

        # 情况 B: 正常 PLAY 阶段
        # 1. SHOW 动作
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                combo = self._evaluate_combo(hand_vals[start:start+length])
                if combo and self._beats(combo, table_combo):
                    actions.append(self.OFFSET_SHOW + self._encode_show(start, length))

        # 2. SCOUT 动作 (桌面有牌且不是自己的)
        # 【修复】：必须限制 n < self.max_hand_size，否则动作编码会越界碰撞！
        if self.table_cards and self.table_owner != player and n < self.max_hand_size:
            sides = [0, 1] if len(self.table_cards) > 1 else [0]
            for side in sides:
                for ins in range(n + 1):
                    for flip in [0, 1]:
                        code = self._encode_scout(side, ins, flip)
                        actions.append(self.OFFSET_SCOUT + code)
                        if self.scout_show_tokens[player]:
                            actions.append(self.OFFSET_SCOUT_SHOW + code)
        if not actions:
            # 这种情况下，在 Scout 规则里通常意味着游戏应该结束
            # 或者我们强制给一个 PASS 动作让游戏轮转下去
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
        # 1. 有人打光手牌
        for i in range(self.num_players):
            if len(self.hands[i]) == 0:
                self.done = True
                self._calculate_scores()
                return
        
        # 2. 轮到某人时，桌上仍是他出的牌（说明没人能压过他）
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

    def get_state(self):
        return {
            "hands": {i: self._get_active_hand(i) for i in range(self.num_players)},
            "table_cards": [c[0] for c in self.table_cards],
            "table_owner": self.table_owner,
            "round_scores": self.round_scores,
            "phase": self.phase
        }