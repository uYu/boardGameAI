#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include "patch_data.h"

// 使用 GCC/Clang 的 __int128 处理 81 位棋盘 (9x9)
typedef __int128 uint128;

struct Action {
    int type;      // 0: MOVE, 1: BUY, 2: PLACE_LEATHER
    int offset;    // 购买时的偏移量 (0, 1, 2)
    uint128 mask;  // 放置的位掩码
    int piece_id;  // 对应的拼块索引
};

struct Player {
    uint128 board = 0;
    int buttons = 5;
    int time_pos = 0;
    int income = 0;
    bool has_7x7_bonus = false;

    inline int count_filled() const {
        return __builtin_popcountll((unsigned long long)board) + 
               __builtin_popcountll((unsigned long long)(board >> 64));
    }

    int get_score() const {
        int empty = 81 - count_filled();
        return buttons - (2 * empty) + (has_7x7_bonus ? 7 : 0);
    }
};

class PatchworkGame {
public:
    Player players[2];
    std::vector<int> pieces_circle; 
    int current_player_idx = 0;
    int token_idx = 0;
    bool pending_leather = false;
    bool game_over = false;
    bool leather_claimed[5] = {false, false, false, false, false};
    
    // 全局抢占标记：确保 7x7 奖赏只有一份
    bool bonus_claimed = false;

    const int income_triggers[8] = {5, 11, 17, 23, 29, 35, 41, 47};
    const int leather_triggers[5] = {20, 26, 32, 44, 50};
    
    static std::vector<uint128> reward_7x7_masks;

    PatchworkGame(const std::vector<PieceData>& all_pieces_data) {
        for(int i = 0; i < (int)all_pieces_data.size(); ++i) {
            pieces_circle.push_back(i);
        }
        if (reward_7x7_masks.empty()) init_static_masks();
        bonus_claimed = false;
    }

    static void init_static_masks() {
        reward_7x7_masks.clear();
        for (int r = 0; r <= 2; ++r) {
            for (int c = 0; c <= 2; ++c) {
                uint128 m = 0;
                for (int dr = 0; dr < 7; ++dr) {
                    for (int dc = 0; dc < 7; ++dc) {
                        m |= ((uint128)1 << ((r + dr) * 9 + (c + dc)));
                    }
                }
                reward_7x7_masks.push_back(m);
            }
        }
    }

    Player& curr() { return players[current_player_idx]; }
    Player& opp() { return players[1 - current_player_idx]; }

    void apply_action(const Action& act, const std::vector<PieceData>& all_pieces) {
        if (act.type == 2) { 
            curr().board |= act.mask;
            pending_leather = false;
            // 放置补丁后也要检查一次 7x7
            check_7x7_logic();
        } 
        else if (act.type == 0) { 
            int dist = std::max(0, (opp().time_pos - curr().time_pos) + 1);
            if (curr().time_pos + dist > 53) dist = 53 - curr().time_pos;
            curr().buttons += dist;
            advance_time(curr(), dist);
        } 
        else if (act.type == 1) { 
            const PieceData& pData = all_pieces[act.piece_id];
            curr().buttons -= pData.cost;
            curr().income += pData.income;
            curr().board |= act.mask;
            
            int circle_pos = (token_idx + act.offset) % pieces_circle.size();
            pieces_circle.erase(pieces_circle.begin() + circle_pos);
            if (!pieces_circle.empty()) {
                token_idx = circle_pos % pieces_circle.size();
            }
            advance_time(curr(), pData.time);
            check_7x7_logic();
        }

        post_action_check();
    }

    std::vector<Action> get_legal_actions(const std::vector<PieceData>& all_pieces) const {
        std::vector<Action> actions;
        const Player& p = players[current_player_idx];

        if (pending_leather) {
            for (int i = 0; i < 81; ++i) {
                uint128 m = ((uint128)1 << i);
                if (!(p.board & m)) actions.push_back({2, 0, m, -1});
            }
            return actions;
        }

        actions.push_back({0, 0, 0, -1});

        int n = pieces_circle.size();
        for (int i = 0; i < std::min(3, n); ++i) {
            int p_idx = pieces_circle[(token_idx + i) % n];
            const PieceData& pData = all_pieces[p_idx];
            if (p.buttons >= pData.cost) {
                for (uint128 m : pData.all_legal_masks) {
                    if (!(p.board & m)) {
                        actions.push_back({1, i, m, p_idx});
                    }
                }
            }
        }
        return actions;
    }

private:
    void check_7x7_logic() {
        // 如果已经有人领过了，不需要再算
        if (bonus_claimed) return;

        for (uint128 m : reward_7x7_masks) {
            if ((curr().board & m) == m) {
                curr().has_7x7_bonus = true;
                bonus_claimed = true; // 锁定奖项
                break;
            }
        }
    }

    void advance_time(Player& p, int steps) {
        int old_pos = p.time_pos;
        p.time_pos = std::min(old_pos + steps, 53);

        for (int trigger : income_triggers) {
            if (old_pos < trigger && p.time_pos >= trigger) {
                p.buttons += p.income;
            }
        }
        for (int i = 0; i < 5; ++i) {
            if (!leather_claimed[i] && old_pos < leather_triggers[i] && p.time_pos >= leather_triggers[i]) {
                leather_claimed[i] = true;
                pending_leather = true; 
                // 注意：由于 pending_leather 会中断当前回合，通常一次只触发一个补丁
                break; 
            }
        }
    }

    void post_action_check() {
        if (pending_leather) return;

        // 时间落后者行动。如果时间相同，则非当前活跃玩家行动（模拟“堆叠”在上方的先动）
        if (players[0].time_pos < players[1].time_pos) {
            current_player_idx = 0;
        } else if (players[1].time_pos < players[0].time_pos) {
            current_player_idx = 1;
        } else {
            // 时间相同时，当前玩家不切换（保持“在上方的玩家”继续动的逻辑，或根据具体规则微调）
        }

        if (players[0].time_pos >= 53 && players[1].time_pos >= 53) {
            game_over = true;
        }
    }
};

// 在全局作用域初始化静态变量
std::vector<uint128> PatchworkGame::reward_7x7_masks;