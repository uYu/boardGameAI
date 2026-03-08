#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include "patchworkGame.hpp"
#include "patch_data.h"
#include "SmartPatchworkAI.hpp"

// --- 玩家类型枚举 ---
enum PlayerType { HUMAN = 1, AI = 2, RANDOM = 3 };

// --- 辅助工具函数 ---

uint128 normalize_mask(uint128 mask) {
    if (mask == 0) return 0;
    int min_r = 9, min_c = 9;
    for (int i = 0; i < 81; ++i) {
        if ((mask >> i) & 1) {
            min_r = std::min(min_r, i / 9);
            min_c = std::min(min_c, i % 9);
        }
    }
    uint128 normal = 0;
    for (int i = 0; i < 81; ++i) {
        if ((mask >> i) & 1) {
            int r = i / 9 - min_r;
            int c = i % 9 - min_c;
            normal |= ((uint128)1 << (r * 9 + c));
        }
    }
    return normal;
}

void print_shape_small(uint128 mask) {
    uint128 normal = normalize_mask(mask);
    for (int r = 0; r < 5; ++r) {
        std::cout << "      ";
        for (int c = 0; c < 5; ++c) {
            if ((normal >> (r * 9 + c)) & 1) std::cout << "\033[1;32m# \033[0m";
            else std::cout << ". ";
        }
        std::cout << std::endl;
    }
}

void print_interactive_board(uint128 board, std::string label) {
    std::cout << "\n  [ " << label << " ]" << std::endl;
    std::cout << "    0 1 2 3 4 5 6 7 8" << std::endl;
    std::cout << "    -----------------" << std::endl;
    for (int r = 0; r < 9; ++r) {
        std::cout << r << " | ";
        for (int c = 0; c < 9; ++c) {
            if ((board >> (r * 9 + c)) & 1) std::cout << "\033[1;32m# \033[0m"; 
            else std::cout << ". ";
        }
        std::cout << std::endl;
    }
}

void print_status(const PatchworkGame& game, const std::vector<PlayerType>& types) {
    std::cout << "\n\033[1;33m" << "==================== 战局状态 ====================" << "\033[0m" << std::endl;
    for (int i = 0; i < 2; ++i) {
        std::string t_name = (types[i] == HUMAN ? "人" : (types[i] == AI ? "AI" : "随机"));
        std::string p_name = "玩家 " + std::to_string(i) + " [" + t_name + "]";
        if (game.current_player_idx == i) p_name = "\033[1;32m-> " + p_name + "\033[0m";
        else p_name = "   " + p_name;
        
        std::cout << p_name << " | 纽扣: " << std::setw(2) << game.players[i].buttons 
                  << " | 收入: " << std::setw(2) << game.players[i].income 
                  << " | 时间: " << std::setw(2) << game.players[i].time_pos << "/53" 
                  << (game.players[i].has_7x7_bonus ? " [7x7奖励]" : "") << std::endl;
    }
}

// --- 主程序 ---

int main() {
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    const std::vector<PieceData>& pieces_pool = ALL_PIECES;
    PatchworkGame game(pieces_pool);
    std::shuffle(game.pieces_circle.begin(), game.pieces_circle.end(), rng);

    // 1. 玩家类型选择菜单
    std::vector<PlayerType> player_types(2);
    for (int i = 0; i < 2; ++i) {
        int choice;
        std::cout << "\033[1;36m请选择 玩家 " << i << " 的类型:\033[0m\n";
        std::cout << "1. 人类玩家 (Human)\n";
        std::cout << "2. 高性能 AI (MCTS)\n";
        std::cout << "3. 随机策略 (Random)\n";
        std::cout << "输入编号 (1-3): ";
        std::cin >> choice;
        player_types[i] = static_cast<PlayerType>(choice);
    }

    SmartPatchworkAI ai_engine(1.0, 60);

    while (!game.game_over) {
        print_status(game, player_types);
        int cur_idx = game.current_player_idx;
        PlayerType cur_type = player_types[cur_idx];

        Action selected_act;

        if (cur_type == AI) {
            std::cout << "\n[AI 正在思考...]" << std::endl;
            selected_act = ai_engine.get_best_action(game, pieces_pool);
            game.apply_action(selected_act, pieces_pool);
        } 
        else if (cur_type == RANDOM) {
            auto actions = game.get_legal_actions(pieces_pool);
            std::uniform_int_distribution<int> dist(0, actions.size() - 1);
            selected_act = actions[dist(rng)];
            std::cout << "\n[随机玩家] 选择了动作。" << std::endl;
            game.apply_action(selected_act, pieces_pool);
        }
        else if (cur_type == HUMAN) {
            // 人类玩家特有 UI
            print_interactive_board(game.players[cur_idx].board, "你的棋盘");
            auto actions = game.get_legal_actions(pieces_pool);

            if (game.pending_leather) {
                std::cout << "\033[1;35m[!] 放置皮革补丁: \033[0m输入坐标 (行 列): ";
                int r, c; std::cin >> r >> c;
                uint128 m = ((uint128)1 << (r * 9 + c));
                bool valid = false;
                for(auto& a : actions) if(a.type == 2 && a.mask == m) { game.apply_action(a, pieces_pool); valid = true; break; }
                if(!valid) std::cout << "无效坐标！" << std::endl;
                continue;
            }

            std::cout << "\n\033[1;32m>> 请选择拼块 <<\033[0m" << std::endl;
            std::vector<int> candidates;
            for (int i = 0; i < std::min(3, (int)game.pieces_circle.size()); ++i) {
                int p_idx = game.pieces_circle[(game.token_idx + i) % game.pieces_circle.size()];
                candidates.push_back(p_idx);
                const auto& p = pieces_pool[p_idx];
                std::cout << "\n选项 [" << i << "] ID:" << p.id << " (纽扣:" << p.cost << " 时间:" << p.time << " 收入:" << p.income << ")" << std::endl;
                print_shape_small(p.all_legal_masks[0]);
            }
            std::cout << "\n选项 [m] 前进" << std::endl;
            std::cout << "输入: ";
            std::string input; std::cin >> input;

            if (input == "m") {
                for(auto& a : actions) if(a.type == 0) { game.apply_action(a, pieces_pool); break; }
            } else {
                int c_idx = std::stoi(input);
                int selected_id = candidates[c_idx];
                std::vector<Action> piece_moves;
                for(auto& a : actions) if(a.type == 1 && a.piece_id == selected_id) piece_moves.push_back(a);

                if(piece_moves.empty()) { std::cout << "无法购买！\n"; continue; }

                std::vector<uint128> unique_shapes;
                for(auto& a : piece_moves) {
                    uint128 s = normalize_mask(a.mask);
                    if(std::find(unique_shapes.begin(), unique_shapes.end(), s) == unique_shapes.end()) unique_shapes.push_back(s);
                }

                std::cout << "\n选择姿态编号:\n";
                for(int i=0; i<unique_shapes.size(); ++i) { std::cout << "姿态 [" << i << "]:\n"; print_shape_small(unique_shapes[i]); }
                int s_idx; std::cin >> s_idx;
                std::cout << "输入坐标 (行 列): ";
                int tr, tc; std::cin >> tr >> tc;

                bool success = false;
                for(auto& a : piece_moves) {
                    if (normalize_mask(a.mask) == unique_shapes[s_idx] && ((a.mask >> (tr * 9 + tc)) & 1)) {
                        game.apply_action(a, pieces_pool); success = true; break;
                    }
                }
                if(!success) std::cout << "放置失败。\n";
            }
        }

        // 如果不是人类回合，打印一下当前版图状态以便观察
        if (cur_type != HUMAN) {
            print_interactive_board(game.players[cur_idx].board, "玩家 " + std::to_string(cur_idx) + " 动作后的版图");
        }
    }

    // --- 最终结算 ---
// --- 详细结算统计 ---
    std::cout << "\n\033[1;32m================== 最终结算明细 ==================\033[0m" << std::endl;
    for (int i = 0; i < 2; ++i) {
        Player& p = game.players[i];
        int filled = p.count_filled();
        int empty = 81 - filled;
        int button_score = p.buttons;
        int penalty = -2 * empty;
        int bonus = p.has_7x7_bonus ? 7 : 0;
        int final_score = p.get_score();

        std::cout << "玩家 " << i << (i == 0 ? " (AI)" : " (Random)") << ":" << std::endl;
        std::cout << "  [+] 纽扣结余: " << std::setw(3) << button_score << std::endl;
        std::cout << "  [+] 7x7 奖励: " << std::setw(3) << bonus << (p.has_7x7_bonus ? " (已获得)" : " (未获得)") << std::endl;
        std::cout << "  [-] 空格惩罚: " << std::setw(3) << penalty << " (" << empty << " 个空格)" << std::endl;
        std::cout << "  -----------------------" << std::endl;
        std::cout << "  \033[1;33m最终得分: " << std::setw(3) << final_score << "\033[0m\n" << std::endl;
    }

    if (game.players[0].get_score() > game.players[1].get_score()) {
        std::cout << "\033[1;36m结论: MCTS AI 胜出！\033[0m" << std::endl;
    } else if (game.players[0].get_score() < game.players[1].get_score()) {
        std::cout << "\033[1;31m结论: 随机策略获胜！\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;33m结论: 平局！\033[0m" << std::endl;
    }

    return 0;
}