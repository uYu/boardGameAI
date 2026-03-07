#include "AzulGame.hpp"
#include "SmartAzulAI.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <thread>
#include <chrono>
#include <regex>

// ==========================================
// 1. 核心视觉样式
// ==========================================
namespace UI {
    const std::string RESET = "\033[0m";
    const std::string BOLD  = "\033[1m";
    const std::string DIM   = "\033[2m";
    const std::string CLEAR = "\033[H\033[J";
    const std::string CYAN  = "\033[36m";
    const std::string GOLD  = "\033[38;5;220m";

    std::string get_color(int color) {
        switch (color) {
            case BLUE:   return "\033[38;5;39m";
            case YELLOW: return "\033[38;5;220m";
            case RED:    return "\033[38;5;196m";
            case BLACK:  return "\033[38;5;244m";
            case WHITE:  return "\033[38;5;255m";
            case FIRST_PLAYER: return "\033[38;5;201m";
            default: return "";
        }
    }

    std::string get_bg(int color) {
        switch (color) {
            case BLUE:   return "\033[48;5;39m";
            case YELLOW: return "\033[48;5;220m";
            case RED:    return "\033[48;5;196m";
            case BLACK:  return "\033[48;5;244m";
            case WHITE:  return "\033[48;5;255m";
            default: return "";
        }
    }

    std::string tile(int color, const std::string& mode = "normal") {
        if (color == EMPTY) return DIM + "◌" + RESET;
        if (color == FIRST_PLAYER) return get_color(color) + "S" + RESET;
        if (mode == "wall_empty") return get_color(color) + "▢" + RESET;
        if (mode == "wall_filled") return get_bg(color) + "  " + RESET;
        return get_color(color) + "■" + RESET;
    }
}

// ==========================================
// 2. 渲染引擎 (修复了数组访问逻辑)
// ==========================================
class AzulRenderer {
    int visible_length(const std::string& s) {
        std::regex ansi_escape("\033\\[[0-9;]*m");
        std::string clean = std::regex_replace(s, ansi_escape, "");
        int len = 0;
        for (size_t i = 0; i < clean.length(); ) {
            if ((clean[i] & 0x80) == 0) { len += 1; i += 1; } 
            else if ((clean[i] & 0xE0) == 0xC0) { len += 1; i += 2; }
            else if ((clean[i] & 0xF0) == 0xE0) { len += 1; i += 3; } 
            else { len += 1; i += 4; }
        }
        return len;
    }

    std::string pad(const std::string& str, int target_len) {
        int v_len = visible_length(str);
        if (v_len < target_len) return str + std::string(target_len - v_len, ' ');
        return str;
    }

    void draw_player(const PlayerBoard& p, bool is_me, std::vector<std::string>& lines) {
        std::string tag = is_me ? (UI::BOLD + "● YOU" + UI::RESET) : (UI::DIM + "○ AI " + UI::RESET);
        lines.push_back(" " + tag + "  Score: " + UI::BOLD + std::to_string(p.score) + UI::RESET);

        for (int r = 0; r < WALL_SIZE; ++r) {
            const auto& line = p.pattern_lines[r];
            std::string pattern_str = "";
            for (int k = 0; k < line.capacity; ++k) {
                if (k >= (line.capacity - line.count)) pattern_str += UI::tile(line.color) + " ";
                else pattern_str += UI::DIM + "·" + UI::RESET + " ";
            }
            if(!pattern_str.empty()) pattern_str.pop_back();

            std::string padding(10 - (line.capacity * 2 - 1), ' ');

            std::string wall_str = "";
            for (int c = 0; c < WALL_SIZE; ++c) {
                int val = p.wall[r][c];
                // 修复点：调用补全后的静态方法 get_wall_color
                if (val == EMPTY) wall_str += UI::tile(PlayerBoard::get_wall_color(r, c), "wall_empty") + " ";
                else wall_str += UI::tile(val, "wall_filled") + " ";
            }
            lines.push_back("  " + std::to_string(r + 1) + " " + padding + pattern_str + " │ " + wall_str);
        }

        // 修复点：floor_line 是数组，需用 floor_count 遍历
        std::string f_tiles = "";
        for (int i = 0; i < p.floor_count; ++i) {
            f_tiles += UI::tile(p.floor_line[i]) + " ";
        }
        lines.push_back("    Floor: [" + (f_tiles.empty() ? " " : f_tiles) + "]");
    }

public:
    void render(const AzulGame& game, int human_idx, const std::vector<std::string>& menu) {
        std::vector<std::string> board;
        board.push_back(" " + UI::BOLD + "MARKET" + UI::RESET);

        // 修复点：center 是数组，需用 center_count 遍历
        std::string c_str = "";
        for (int i = 0; i < game.center_count; ++i) {
            c_str += UI::tile(game.center[i]) + " ";
        }
        board.push_back(" [C] Center: " + c_str);

        for (int i = 0; i < game.num_factories; i += 2) {
            std::string f1 = "";
            for (int j = 0; j < 4; ++j) {
                if (game.factories[i][j] != EMPTY) f1 += UI::tile(game.factories[i][j]) + " ";
            }
            std::string row = " [" + std::to_string(i + 1) + "] " + pad(f1, 12);
            if (i + 1 < game.num_factories) {
                std::string f2 = "";
                for (int j = 0; j < 4; ++j) {
                    if (game.factories[i+1][j] != EMPTY) f2 += UI::tile(game.factories[i+1][j]) + " ";
                }
                row += " [" + std::to_string(i + 2) + "] " + f2;
            }
            board.push_back(row);
        }

        board.push_back(" " + std::string(36, '-'));

        std::vector<int> p_ids = {1 - human_idx, human_idx};
        for (int pid : p_ids) {
            draw_player(game.players[pid], pid == human_idx, board);
            board.push_back("");
        }

        std::cout << UI::CLEAR;
        std::cout << " ┌" << std::string(44, '-') << "┬" << std::string(38, '-') << "┐\n";

        int max_lines = (int)std::max(board.size(), menu.size());
        for (int i = 0; i < max_lines; ++i) {
            std::string b = (i < (int)board.size()) ? board[i] : "";
            std::string m = (i < (int)menu.size()) ? menu[i] : "";
            std::cout << " │ " << pad(b, 42) << " │ " << pad(m, 36) << " │\n";
        }
        std::cout << " └" << std::string(44, '-') << "┴" << std::string(38, '-') << "┘\n";
    }
};

// Wizard 类和 main 函数中对 UI 的调用逻辑保持基本一致，但需确保与补全后的 AzulGame 接口匹配
// ...（Wizard类代码基本不变，主要在主循环中做少量修正）...

class Wizard {
    AzulRenderer& renderer;
public:
    std::vector<AzulGame> history;
    Wizard(AzulRenderer& r) : renderer(r) {}
    void save_state(const AzulGame& game) { history.push_back(game); }

    std::string get_input(const std::string& prompt) {
        std::string ans;
        std::cout << "  " << UI::CYAN << prompt << " > " << UI::RESET;
        std::cin >> ans;
        for (auto& c : ans) c = toupper(c);
        return ans;
    }

    bool ask(const AzulGame& game, int human_idx, Action& out_action) {
        std::vector<std::string> logs = {
            " " + UI::GOLD + "◆ ROUND " + std::to_string(game.round_number) + " ◆" + UI::RESET,
            " " + UI::BOLD + "ACTION MENU" + UI::RESET,
            " " + UI::DIM + "(Type 'U' to Undo)" + UI::RESET, 
            ""
        };

        auto legal = game.get_legal_actions();
        std::set<int> srcs;
        for (const auto& a : legal) srcs.insert(a.source_id);

        logs.push_back(" " + UI::BOLD + "1. Select Source:" + UI::RESET);
        std::map<std::string, int> s_map;
        for (int s : srcs) {
            if (s != -1) {
                s_map[std::to_string(s + 1)] = s;
                logs.push_back("  (" + std::to_string(s + 1) + ") Factory " + std::to_string(s + 1));
            }
        }
        if (srcs.count(-1)) { s_map["C"] = -1; logs.push_back("  (C) Center"); }

        int sel_s;
        while (true) {
            renderer.render(game, human_idx, logs);
            std::string res = get_input("Source ID / U");
            if (res == "U") return false;
            if (s_map.count(res)) { sel_s = s_map[res]; break; }
        }

        logs.resize(4);
        logs.push_back(" " + UI::DIM + "Source: " + (sel_s == -1 ? "Center" : "F" + std::to_string(sel_s + 1)) + UI::RESET);
        logs.push_back("");
        logs.push_back(" " + UI::BOLD + "2. Select Color:" + UI::RESET);

        std::set<int> cols;
        for (const auto& a : legal) if (a.source_id == sel_s) cols.insert(a.color);
        std::map<std::string, int> c_map;
        int idx = 1;
        for (int c : cols) {
            c_map[std::to_string(idx)] = c;
            logs.push_back("  (" + std::to_string(idx) + ") " + UI::tile(c));
            idx++;
        }

        int sel_c;
        while (true) {
            renderer.render(game, human_idx, logs);
            std::string res = get_input("Color Index / U");
            if (res == "U") return false;
            if (c_map.count(res)) { sel_c = c_map[res]; break; }
        }

        logs.resize(7);
        logs.push_back(" " + UI::DIM + "Color: " + UI::tile(sel_c) + UI::RESET);
        logs.push_back("");
        logs.push_back(" " + UI::BOLD + "3. Select Row:" + UI::RESET);

        std::map<std::string, Action> r_map;
        for (const auto& a : legal) {
            if (a.source_id == sel_s && a.color == sel_c) {
                std::string key = (a.target_row == -1) ? "F" : std::to_string(a.target_row + 1);
                r_map[key] = a;
                std::string name = (a.target_row == -1) ? "Floor" : "Row " + std::to_string(a.target_row + 1);
                logs.push_back("  (" + key + ") " + name);
            }
        }

        while (true) {
            renderer.render(game, human_idx, logs);
            std::string res = get_input("Row ID / U");
            if (res == "U") return false;
            if (r_map.count(res)) { out_action = r_map[res]; return true; }
        }
    }
};

int main() {
    std::cout << UI::CLEAR << "\n " << UI::BOLD << "WELCOME TO AZUL (C++ Edition)" << UI::RESET << "\n\n";
    std::cout << " " << UI::GOLD << "--- GAME SETTINGS ---" << UI::RESET << "\n\n";

    int human_idx = 0;
    std::string choice;
    std::cout << " 1. Who goes first?\n    [1] Human\n    [2] AI\n  > ";
    std::cin >> choice;
    if (choice == "2") human_idx = 1;

    double ai_time = 1.0;
    std::cout << "\n 2. AI Strength:\n    [1] Fast\n    [2] Normal\n    [3] Deep\n  > ";
    std::cin >> choice;
    if (choice == "1") ai_time = 0.5; else if (choice == "3") ai_time = 1.5;

    std::cin.ignore(10000, '\n');
    std::cout << "\n  Press Enter to start...";
    std::cin.get();

    AzulGame game(2);
    AzulRenderer renderer;
    Wizard wizard(renderer);
    SmartAzulAI ai(ai_time, 50);

    while (!game.game_over) {
        if (game.current_player_idx == human_idx) {
            wizard.save_state(game);
            Action action;
            if (!wizard.ask(game, human_idx, action)) {
                if (wizard.history.size() >= 2) {
                    wizard.history.pop_back(); 
                    game = wizard.history.back();
                    wizard.history.pop_back();
                    continue;
                } else {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
            }
            game.step(action);
        } else {
            renderer.render(game, human_idx, {
                " " + UI::GOLD + "◆ ROUND " + std::to_string(game.round_number) + " ◆" + UI::RESET,
                "", " " + UI::CYAN + "AI is thinking (MCTS)..." + UI::RESET
            });
            Action action = ai.get_best_action(game);
            game.step(action);
        }

        bool round_ended = false;
        for (int i = 0; i < 2; ++i) if (game.players[i].round_just_finished) round_ended = true;

        if (round_ended) {
            renderer.render(game, human_idx, {"", " Round Finished!", " Scoring..."});
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    renderer.render(game, human_idx, {"", " " + UI::GOLD + "GAME OVER!" + UI::RESET});
    game.print_final_stats();
    return 0;
}