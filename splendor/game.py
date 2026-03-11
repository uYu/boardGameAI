import random
from typing import List, Dict, Optional
# ================= 游戏常量与数据 =================
COLORS = ['white', 'blue', 'green', 'red', 'black']
ALL_COLORS_GOLD = COLORS + ['gold']

# 简化存储：[颜色, 分数, {消耗}]
RAW_CARDS_L1 = [
    # 蓝色奖励卡
    ["blue", 0, {"black": 3}], ["blue", 0, {"white": 1, "green": 1, "red": 1, "black": 1}],
    ["blue", 0, {"white": 1, "green": 2, "red": 1, "black": 1}], ["blue", 0, {"white": 1, "black": 2}],
    ["blue", 1, {"green": 4}], ["blue", 0, {"white": 1, "green": 1, "black": 1}],
    ["blue", 0, {"green": 1, "red": 2, "black": 1}], ["blue", 0, {"white": 2, "blue": 0, "green": 0, "red": 0, "black": 1}],
    # 白色奖励卡
    ["white", 0, {"blue": 3}], ["white", 0, {"blue": 1, "green": 1, "red": 1, "black": 1}],
    ["white", 0, {"blue": 2, "green": 1}], ["white", 0, {"blue": 2, "green": 2, "black": 1}],
    ["white", 1, {"red": 4}], ["white", 0, {"blue": 1, "green": 2, "red": 1, "black": 1}],
    ["white", 0, {"red": 2, "black": 1}], ["white", 0, {"white": 0, "blue": 1, "green": 1, "red": 1, "black": 1}],
    # 绿色奖励卡
    ["green", 0, {"red": 3}], ["green", 0, {"white": 1, "blue": 1, "red": 1, "black": 1}],
    ["green", 1, {"black": 4}], ["green", 0, {"white": 1, "blue": 1, "red": 2, "black": 1}],
    ["green", 0, {"white": 1, "blue": 1, "red": 1}], ["green", 0, {"white": 2, "blue": 1}],
    ["green", 0, {"white": 1, "blue": 2, "red": 1, "black": 1}], ["green", 0, {"white": 1, "blue": 0, "green": 0, "red": 1, "black": 2}],
    # 红色奖励卡
    ["red", 0, {"white": 3}], ["red", 0, {"white": 1, "blue": 1, "green": 1, "black": 1}],
    ["red", 1, {"white": 4}], ["red", 0, {"white": 2, "blue": 1, "green": 1, "black": 1}],
    ["red", 0, {"white": 2, "red": 0, "black": 2}], ["red", 0, {"white": 1, "blue": 1, "green": 1}],
    ["red", 0, {"white": 1, "blue": 2, "green": 1, "black": 1}], ["red", 0, {"blue": 2, "green": 1}],
    # 黑色奖励卡
    ["black", 0, {"green": 3}], ["black", 0, {"white": 1, "blue": 1, "green": 1, "red": 1}],
    ["black", 1, {"blue": 4}], ["black", 0, {"white": 1, "blue": 2, "green": 1, "red": 1}],
    ["black", 0, {"white": 1, "blue": 1, "red": 1}], ["black", 0, {"blue": 1, "green": 2, "red": 1}],
    ["black", 0, {"white": 2, "blue": 1, "green": 2}], ["black", 0, {"green": 2, "red": 1}]
]

RAW_CARDS_L2 = [
    # 蓝色
    ["blue", 1, {"blue": 2, "green": 2, "red": 3}], ["blue", 2, {"blue": 5}],
    ["blue", 2, {"white": 5, "blue": 3}], ["blue", 1, {"white": 2, "red": 2, "black": 3}],
    ["blue", 2, {"white": 2, "black": 3, "red": 2}], ["blue", 3, {"blue": 6}],
    # 绿色
    ["green", 1, {"white": 2, "blue": 3, "black": 2}], ["green", 2, {"green": 5}],
    ["green", 2, {"white": 4, "blue": 2, "green": 1}], ["green", 1, {"white": 3, "green": 2, "red": 3}],
    ["green", 2, {"blue": 5, "green": 3}], ["green", 3, {"green": 6}],
    # 红色
    ["red", 1, {"white": 2, "red": 2, "black": 3}], ["red", 2, {"red": 5}],
    ["red", 2, {"blue": 3, "green": 2, "red": 3}], ["red", 1, {"blue": 3, "green": 2, "black": 2}],
    ["red", 2, {"green": 5, "red": 3}], ["red", 3, {"red": 6}],
    # 黑色
    ["black", 1, {"white": 3, "blue": 2, "green": 2}], ["black", 2, {"black": 5}],
    ["black", 2, {"white": 3, "blue": 2, "green": 2}], ["black", 1, {"white": 3, "blue": 3, "black": 2}],
    ["black", 2, {"red": 5, "black": 3}], ["black", 3, {"black": 6}],
    # 白色
    ["white", 1, {"green": 3, "red": 2, "black": 2}], ["white", 2, {"white": 5}],
    ["white", 2, {"red": 5, "white": 3}], ["white", 1, {"blue": 2, "green": 3, "black": 2}],
    ["white", 2, {"black": 5, "white": 3}], ["white", 3, {"white": 6}]
]

RAW_CARDS_L3 = [
    ["white", 3, {"black": 7}], ["white", 4, {"white": 3, "black": 6, "red": 3}],
    ["white", 4, {"black": 7}], ["white", 5, {"white": 3, "black": 7}],
    ["blue", 3, {"white": 7}], ["blue", 4, {"white": 6, "blue": 3, "green": 3}],
    ["blue", 4, {"white": 7}], ["blue", 5, {"white": 7, "blue": 3}],
    ["green", 3, {"blue": 7}], ["green", 4, {"blue": 6, "green": 3, "red": 3}],
    ["green", 4, {"blue": 7}], ["green", 5, {"blue": 7, "green": 3}],
    ["red", 3, {"green": 7}], ["red", 4, {"green": 6, "red": 3, "black": 3}],
    ["red", 4, {"green": 7}], ["red", 5, {"green": 7, "red": 3}],
    ["black", 3, {"red": 7}], ["black", 4, {"red": 6, "black": 3, "white": 3}],
    ["black", 4, {"red": 7}], ["black", 5, {"red": 7, "black": 3}]
]

NOBLES_DATA = [
    {"points": 3, "req": {"red": 4, "green": 4}},
    {"points": 3, "req": {"white": 4, "blue": 4}},
    {"points": 3, "req": {"blue": 4, "green": 4}},
    {"points": 3, "req": {"black": 4, "red": 4}},
    {"points": 3, "req": {"white": 4, "black": 4}},
    {"points": 3, "req": {"white": 3, "blue": 3, "black": 3}},
    {"points": 3, "req": {"white": 3, "red": 3, "black": 3}},
    {"points": 3, "req": {"blue": 3, "green": 3, "red": 3}},
    {"points": 3, "req": {"white": 3, "blue": 3, "green": 3}},
    {"points": 3, "req": {"green": 3, "red": 3, "black": 3}},
]

# ================= 游戏常量 =================
COLORS = ['white', 'blue', 'green', 'red', 'black']
ALL_COLORS_GOLD = COLORS + ['gold']

class Card:
    def __init__(self, level: int, color: str, points: int, cost: Dict[str, int]):
        self.level = level
        self.color = color
        self.points = points
        self.cost = {c: cost.get(c, 0) for c in COLORS}

class Noble:
    def __init__(self, points: int, req: Dict[str, int]):
        self.points = points
        self.requirement = {c: req.get(c, 0) for c in COLORS}

class Player:
    def __init__(self, pid: int):
        self.id = pid
        self.gems = {c: 0 for c in ALL_COLORS_GOLD}
        self.bonuses = {c: 0 for c in COLORS}
        self.reserved: List[Card] = []
        self.purchased: List[Card] = []
        self.nobles: List[Noble] = []
        self.score = 0

    def total_gems(self):
        return sum(self.gems.values())

    def can_afford(self, card: Card) -> bool:
        gold_needed = 0
        for c in COLORS:
            needed = max(0, card.cost[c] - self.bonuses[c])
            if self.gems[c] < needed:
                gold_needed += (needed - self.gems[c])
        return self.gems['gold'] >= gold_needed

class SplendorGame:
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        # 2人局宝石初值为4，黄金为5
        gem_init = {2: 4, 3: 5, 4: 7}[num_players]
        self.bank = {c: gem_init for c in COLORS}
        self.bank['gold'] = 5
        
        self.players = [Player(i) for i in range(num_players)]
        self.current_idx = 0
        self.turn_count = 0  # 记录总步数
        
        # 数据导入与洗牌 (此处省略 RAW_CARDS 数据定义，假设已在外部定义)
        self.decks = {
            1: [Card(1, *d) for d in RAW_CARDS_L1],
            2: [Card(2, *d) for d in RAW_CARDS_L2],
            3: [Card(3, *d) for d in RAW_CARDS_L3]
        }
        for l in [1, 2, 3]: random.shuffle(self.decks[l])
        
        # 场面初始化
        self.board = {l: [self.decks[l].pop() if self.decks[l] else None for _ in range(4)] for l in [1,2,3]}
        self.nobles = [Noble(n['points'], n['req']) for n in random.sample(NOBLES_DATA, num_players + 1)]

    # --- 核心动作 ---

    def take_gems(self, colors: List[str], count: int = 1) -> bool:
        """
        拿取宝石动作。
        count=1 时，colors 为 3 种不同颜色；
        count=2 时，colors 为 1 种颜色。
        """
        p = self.players[self.current_idx]
        
        # 1. 验证合法性
        if count == 2:
            color = colors[0]
            if self.bank[color] < 4: return False  # 规则：银行>=4才能拿2个
        else:
            for c in colors:
                if self.bank[c] <= 0: return False

        # 2. 执行拿取
        for c in colors:
            self.bank[c] -= count
            p.gems[c] += count
            
        self._handle_gem_limit(p)
        return True

    def buy_card(self, level: int, idx: int, from_reserved: bool = False) -> bool:
        """购买卡牌"""
        p = self.players[self.current_idx]
        
        if from_reserved:
            if idx >= len(p.reserved): return False
            card = p.reserved[idx]
        else:
            if level not in self.board or idx >= 4: return False
            card = self.board[level][idx]
        
        if card is None or not p.can_afford(card): return False
        
        # 3. 计算支付并扣除
        gold_needed = 0
        for c in COLORS:
            cost_after_bonus = max(0, card.cost[c] - p.bonuses[c])
            if p.gems[c] >= cost_after_bonus:
                p.gems[c] -= cost_after_bonus
                self.bank[c] += cost_after_bonus
            else:
                shortfall = cost_after_bonus - p.gems[c]
                gold_needed += shortfall
                self.bank[c] += p.gems[c]
                p.gems[c] = 0
        
        p.gems['gold'] -= gold_needed
        self.bank['gold'] += gold_needed
        
        # 4. 更新场面/玩家资产
        if from_reserved:
            p.reserved.pop(idx)
        else:
            self.board[level][idx] = self.decks[level].pop() if self.decks[level] else None
            
        p.purchased.append(card)
        p.bonuses[card.color] += 1
        p.score += card.points
        
        # 5. 检查贵族
        self._check_nobles(p)
        return True

    def reserve_card(self, level: int, idx: int = -1, is_blind: bool = False) -> bool:
        """保留卡牌"""
        p = self.players[self.current_idx]
        if len(p.reserved) >= 3: return False
        
        card = None
        if is_blind:
            if self.decks[level]: card = self.decks[level].pop()
        else:
            if idx < 0 or idx >= 4: return False
            card = self.board[level][idx]
            if card:
                self.board[level][idx] = self.decks[level].pop() if self.decks[level] else None
        
        if card is None: return False
        
        p.reserved.append(card)
        if self.bank['gold'] > 0:
            self.bank['gold'] -= 1
            p.gems['gold'] += 1
            
        self._handle_gem_limit(p)
        return True

    # --- 内部逻辑 ---

    def _handle_gem_limit(self, player: Player):
        """超过10个宝石强制归还。优先归还普通宝石，黄金最后归还。"""
        while player.total_gems() > 10:
            # 找到数量大于0的普通宝石颜色
            eligible = [c for c in COLORS if player.gems[c] > 0]
            if not eligible: # 只有黄金的情况
                c = 'gold'
            else:
                c = random.choice(eligible)
            player.gems[c] -= 1
            self.bank[c] += 1

    def _check_nobles(self, player: Player):
        """规则：每回合最多获得一名贵族。"""
        for i, n in enumerate(self.nobles):
            if all(player.bonuses[c] >= n.requirement[c] for c in COLORS):
                player.score += n.points
                player.nobles.append(self.nobles.pop(i))
                break 

    def is_game_over(self) -> bool:
        """有人达15分，且回到初始玩家（保证每人回合数相同）时结束。"""
        has_reached = any(p.score >= 15 for p in self.players)
        return has_reached and self.current_idx == 0

    def next_turn(self):
        self.current_idx = (self.current_idx + 1) % self.num_players
        self.turn_count += 1