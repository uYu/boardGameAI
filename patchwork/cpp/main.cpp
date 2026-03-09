#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>

#include "patchworkGame.hpp"
#include "patch_data.h"
#include "SmartPatchworkAI.hpp"

// --- 玩家类型枚举 ---
enum PlayerType { HUMAN = 1, AI = 2, RANDOM = 3 };

// --- 1. 完善的辅助工具函数 (找回并优化) ---

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
            if (r >= 0 && r < 9 && c >= 0 && c < 9)
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

void print_dual_boards(const PatchworkGame& game, const std::vector<PlayerType>& types) {
    int cur_idx = game.current_player_idx;
    
    std::cout << "\n";
    // 打印标题头
    for (int i = 0; i < 2; ++i) {
        std::string t_name = (types[i] == HUMAN ? "人类" : (types[i] == AI ? "AI" : "随机"));
        std::string label = "玩家 " + std::to_string(i) + " [" + t_name + "]";
        
        if (i == cur_idx) {
            // 当前行动者用 绿色 + 闪烁箭头 标识
            std::cout << "  \033[1;32m===> " << std::left << std::setw(16) << label << " <===\033[0m   ";
        } else {
            std::cout << "       " << std::left << std::setw(16) << label << "       ";
        }
        if (i == 0) std::cout << "      "; // 两个棋盘中间的间距
    }
    std::cout << "\n    0 1 2 3 4 5 6 7 8              0 1 2 3 4 5 6 7 8" << std::endl;
    std::cout << "    -----------------              -----------------" << std::endl;

    for (int r = 0; r < 9; ++r) {
        // 打印两人的同一行
        for (int p_idx = 0; p_idx < 2; ++p_idx) {
            std::cout << r << " | ";
            uint128 board = game.players[p_idx].board;
            for (int c = 0; c < 9; ++c) {
                if ((board >> (r * 9 + c)) & 1) {
                    // 填充处根据是否为当前玩家显示不同颜色
                    if (p_idx == cur_idx) std::cout << "\033[1;32m# \033[0m"; // 活跃玩家绿色
                    else std::cout << "\033[1;37m# \033[0m";                  // 等待玩家白色
                } else {
                    std::cout << ". ";
                }
            }
            if (p_idx == 0) std::cout << "          "; // 棋盘间的间距
        }
        std::cout << "\n";
    }
    std::cout << "    -----------------              -----------------" << std::endl;
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

// --- 2. 更新后的 Main 主循环 ---

int main() {
    std::vector<PlayerType> player_types(2);
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    
    PatchworkGame game; 
    std::shuffle(game.pieces_circle.begin(), game.pieces_circle.end(), rng);

    // 玩家选择
    std::cout << "选择玩家 0 类型 (1:人类, 2:AI, 3:随机): ";
    int t0; std::cin >> t0; player_types[0] = static_cast<PlayerType>(t0);
    std::cout << "选择玩家 1 类型 (1:人类, 2:AI, 3:随机): ";
    int t1; std::cin >> t1; player_types[1] = static_cast<PlayerType>(t1);

    SmartPatchworkAI ai_engine(4.0, 100);

 while (!game.game_over) {
        print_status(game, player_types);
        
        int cur_idx = game.current_player_idx;
        PlayerType cur_type = player_types[cur_idx];
        auto actions = game.get_legal_actions();

        // --- 新增：在这里统一打印当前轮到谁的棋盘 ---
        // 这样无论是 AI、随机还是人类，你都能看到棋盘状态
        std::string board_label = "玩家 " + std::to_string(cur_idx) + " 的当前棋盘";
        print_dual_boards(game, player_types);
        // ---------------------------------------

        if (actions.empty()) {
            std::cout << "\n[警告] 当前玩家无合法动作，强制跳过。" << std::endl;
            Action force_move = {0, -1, 0, 53 - game.players[cur_idx].time_pos};
            game.apply_action(force_move);
            continue;
        }

        if (cur_type == AI) {
            std::cout << "\n[AI 正在思考...]" << std::endl;
            Action selected_act = ai_engine.get_best_action(game); 
            game.apply_action(selected_act);
        } 
        else if (cur_type == RANDOM) {
            std::uniform_int_distribution<int> dist(0, (int)actions.size() - 1);
            game.apply_action(actions[dist(rng)]);
            std::cout << "\n[随机玩家] 已执行动作。" << std::endl;
            // 适当增加延迟，方便你观察棋盘变化
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
        }
        else if (cur_type == HUMAN) {
            // 处理皮革补丁放置
            if (game.pending_leather) {
                print_dual_boards(game, player_types);
                std::cout << "\033[1;35m[!] 获得皮革！输入坐标 (r c): \033[0m";
                int r, c; std::cin >> r >> c;
                uint128 m = ((uint128)1 << (r * 9 + c));
                bool found = false;
                for(auto& a : actions) if(a.type == 2 && a.mask == m) { game.apply_action(a); found = true; break; }
                if(!found) std::cout << "位置非法！" << std::endl;
                continue;
            }

            print_dual_boards(game, player_types);
            
            // 展示商店选项
            std::cout << "\n--- 可选操作 ---" << std::endl;
            std::cout << "[H] 前进并获取纽扣 (移动后领先对手一格)" << std::endl;
            
            int n = (int)game.pieces_circle.size();
            for (int i = 0; i < std::min(3, n); ++i) {
                int p_idx = game.pieces_circle[(game.token_idx + i) % n];
                const PieceData& pData = ALL_PIECES[p_idx];
                std::cout << "[" << i << "] 买补丁 ID:" << pData.id << " (纽扣:" << pData.cost 
                          << " 时间:" << pData.time << " 收入:" << pData.income << ")" << std::endl;
                print_shape_small(GLOBAL_MASK_POOL[pData.mask_offset]);
            }

            std::cout << "请输入选择 (h 或 0/1/2): ";
            char cmd; std::cin >> cmd;
            
            if (cmd == 'h' || cmd == 'H') {
                for(auto& a : actions) if(a.type == 0) { game.apply_action(a); break; }
            } else if (isdigit(cmd)) {
                int choice = cmd - '0';
                // 人类选择后，这里自动尝试放置（简化逻辑：找到第一个匹配该 ID 的合法 mask）
                bool ok = false;
                int target_id = game.pieces_circle[(game.token_idx + choice) % n];
                for(auto& a : actions) {
                    if(a.type == 1 && a.piece_id == target_id) {
                        game.apply_action(a);
                        ok = true;
                        break;
                    }
                }
                if(!ok) std::cout << "无法购买或放置该补丁！" << std::endl;
            }
        }
    }

    // --- 最终结算 ---
    std::cout << "\n\033[1;32m================== 最终结算明细 ==================\033[0m" << std::endl;
    for (int i = 0; i < 2; ++i) {
        Player& p = game.players[i];
        int filled = p.count_filled();
        int empty = 81 - filled;
        std::cout << "玩家 " << i << " [" << (player_types[i] == AI ? "AI" : "其它") << "]:" << std::endl;
        std::cout << "  纽扣: " << p.buttons << " | 7x7奖励: " << (p.has_7x7_bonus ? 7 : 0) << " | 空位惩罚: -" << (2 * empty) << std::endl;
        std::cout << "  \033[1;33m总分: " << p.get_score() << "\033[0m" << std::endl;
    }

    return 0;
}