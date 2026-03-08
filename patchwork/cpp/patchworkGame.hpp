#pragma once
#include <vector>
#include <stdint.h>
#include <algorithm>
#include <iostream>
#include "patch_data.h"

typedef __int128_t uint128;

struct Action {
    int type;      // 0: 前进, 1: 买补丁, 2: 放置皮革
    int piece_id;  // 补丁 ID (ALL_PIECES 的索引)
    uint128 mask;  // 棋盘掩码
    int offset;    // type=0时为移动步数; type=1时为在圆环中的偏移(0,1,2)
};

struct Player {
    uint128 board = 0;
    int buttons = 5;
    int income = 0;
    int time_pos = 0;
    bool has_7x7_bonus = false;

    int count_filled() const {
        int count = 0;
        uint128 b = board;
        while (b) { b &= (b - 1); count++; }
        return count;
    }

    int get_score() const {
        return buttons + (has_7x7_bonus ? 7 : 0) - 2 * (81 - count_filled());
    }
};

class PatchworkGame {
public:
    static constexpr int income_triggers[8] = {5, 11, 17, 23, 29, 35, 41, 47};
    static constexpr int leather_triggers[5] = {20, 26, 32, 44, 50};
    static inline uint128 reward_7x7_masks[9];
    static inline bool static_init_done = false;

    Player players[2];
    int current_player_idx = 0;
    std::vector<int> pieces_circle;
    int token_idx = 0;
    int leather_claimed_mask = 0; 
    bool pending_leather = false;
    bool bonus_claimed = false;
    bool game_over = false;

    static void init_static_masks() {
        if (static_init_done) return;
        for (int r = 0; r <= 2; ++r) {
            for (int c = 0; c <= 2; ++c) {
                uint128 m = 0;
                for (int dr = 0; dr < 7; ++dr) {
                    for (int dc = 0; dc < 7; ++dc) 
                        m |= ((uint128)1 << ((r + dr) * 9 + (c + dc)));
                }
                reward_7x7_masks[r * 3 + c] = m;
            }
        }
        static_init_done = true;
    }

    PatchworkGame() {
        init_static_masks();
        pieces_circle.reserve(33);
        for (int i = 0; i < 33; ++i) pieces_circle.push_back(i);
    }

    std::vector<Action> get_legal_actions() const {
        std::vector<Action> actions;
        const Player& p = players[current_player_idx];
        const Player& opp = players[1 - current_player_idx];

        // 1. 皮革补丁放置阶段
        if (pending_leather) {
            for (int i = 0; i < 81; ++i) {
                if (!(p.board & ((uint128)1 << i))) 
                    actions.push_back({2, -1, (uint128)1 << i, 0});
            }
            return actions;
        }

// 修改 get_legal_actions 里的前进逻辑
    if (p.time_pos < 53) {
    // 即使对手也在 53，玩家也应该能走到 53（移动步数为 53 - p.time_pos）
    int dist = (opp.time_pos >= p.time_pos) ? (opp.time_pos - p.time_pos + 1) : 1;
    if (p.time_pos + dist > 53) dist = 53 - p.time_pos;
    
    if (dist > 0) {
        actions.push_back({0, -1, 0, dist});
    } else if (p.time_pos < 53) {
        // 兜底：如果算出来 dist 是 0 但还没到终点，强制走 1 步
        actions.push_back({0, -1, 0, 1});
    }
}

        // 3. 购买补丁 (仅看前三个)
        int n = pieces_circle.size();
        for (int i = 0; i < std::min(3, n); ++i) {
            int p_idx = pieces_circle[(token_idx + i) % n];
            const PieceData& piece = ALL_PIECES[p_idx];

            if (p.buttons >= piece.cost) {
                for (int m_idx = 0; m_idx < piece.mask_count; ++m_idx) {
                    uint128 m = GLOBAL_MASK_POOL[piece.mask_offset + m_idx];
                    if (!(p.board & m)) actions.push_back({1, p_idx, m, i});
                }
            }
        }
        return actions;
    }

    void apply_action(const Action& act) {
        Player& p = players[current_player_idx];

        if (act.type == 0) { // 前进
            p.buttons += act.offset;
            advance_time(p, act.offset);
        } 
        else if (act.type == 1) { // 购买
            const PieceData& piece = ALL_PIECES[act.piece_id];
            p.buttons -= piece.cost;
            p.board |= act.mask;
            p.income += piece.income;
            
            int remove_pos = (token_idx + act.offset) % pieces_circle.size();
            pieces_circle.erase(pieces_circle.begin() + remove_pos);
            token_idx = pieces_circle.empty() ? 0 : (remove_pos % pieces_circle.size());
            
            check_7x7_bonus(p);
            advance_time(p, piece.time);
        } 
        else if (act.type == 2) { // 放置皮革
            p.board |= act.mask;
            pending_leather = false;
            check_7x7_bonus(p);
        }
        update_turn_and_game_over();
    }

    void advance_time(Player& p, int steps) {
        int old_pos = p.time_pos;
        p.time_pos = std::min(53, p.time_pos + steps);

        for (int trigger : income_triggers) {
            if (old_pos < trigger && p.time_pos >= trigger) p.buttons += p.income;
        }

        for (int i = 0; i < 5; ++i) {
            if (!(leather_claimed_mask & (1 << i)) && 
                old_pos < leather_triggers[i] && p.time_pos >= leather_triggers[i]) {
                leather_claimed_mask |= (1 << i);
                pending_leather = true;
                break; 
            }
        }
    }

    void check_7x7_bonus(Player& p) {
        if (bonus_claimed) return;
        for (int i = 0; i < 9; ++i) {
            if ((p.board & reward_7x7_masks[i]) == reward_7x7_masks[i]) {
                p.has_7x7_bonus = true;
                bonus_claimed = true;
                return;
            }
        }
    }

    void update_turn_and_game_over() {
    if (pending_leather) return; 

    if (players[0].time_pos >= 53 && players[1].time_pos >= 53) {
        game_over = true;
        return;
    }

    int opp_idx = 1 - current_player_idx;
    
    // 只有当对手严格落后于当前玩家时，才换人
    if (players[opp_idx].time_pos < players[current_player_idx].time_pos) {
        current_player_idx = opp_idx;
    }
    // 如果当前玩家仍然落后，或者两人时间相等，则 current_player_idx 不变
    // 注意：如果是当前玩家移动后刚好追平对手，按规则也是当前玩家继续（除非他想让出）
    // 但 Patchwork 规则实际上是“谁在下面谁走”，追平者在上方，所以追平后应该换人。
    // 修正为：
    else if (players[current_player_idx].time_pos > players[opp_idx].time_pos) {
         current_player_idx = opp_idx;
    }
    // 简单的逻辑：谁小谁走。如果相等，则换人。
    if (players[current_player_idx].time_pos >= 53 && players[opp_idx].time_pos < 53) {
        current_player_idx = opp_idx;
    }
}
};