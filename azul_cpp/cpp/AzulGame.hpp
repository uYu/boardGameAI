#pragma once
#ifndef AZUL_GAME_HPP
#define AZUL_GAME_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstring>
#include <sstream>

typedef int8_t Tile;

enum Color : Tile { 
    EMPTY = -1, 
    BLUE = 0, YELLOW = 1, RED = 2, BLACK = 3, WHITE = 4, 
    FIRST_PLAYER = 5 
};

constexpr int8_t NUM_COLORS = 5;
constexpr int8_t WALL_SIZE = 5;
constexpr int8_t MAX_CENTER = 100;
inline const int8_t FLOOR_PENALTIES[] = {-1, -1, -2, -2, -2, -3, -3};

struct Action {
    int8_t source_id; // -1 为中心区, 0-8 为工厂
    int8_t color;     // 0-4 颜色
    int8_t target_row; // 0-4 为修整行, -1 为地板

    bool operator<(const Action& other) const {
        if (source_id != other.source_id) return source_id < other.source_id;
        if (color != other.color) return color < other.color;
        return target_row < other.target_row;
    }
    bool operator==(const Action& other) const {
        return source_id == other.source_id && color == other.color && target_row == other.target_row;
    }
    
    // 方便前端通信的唯一标识符
    std::string id() const {
        return std::to_string(source_id) + "_" + std::to_string(color) + "_" + std::to_string(target_row);
    }
};

struct MoveRecord {
    Action action;
    int8_t player_idx;
    Tile factory_prev_state[4];
    int8_t center_prev_count;
    int8_t first_player_token_owner_prev;
    int32_t prev_score;
    int8_t prev_line_count;
    int8_t prev_line_color;
    Tile prev_floor_line[7];
    int8_t prev_floor_count;
    bool round_ended;
    bool first_player_taken;
};

class PlayerBoard {
public:
    int32_t score;
    Tile wall[WALL_SIZE][WALL_SIZE];
    struct PatternLine {
        Tile color = EMPTY;
        int8_t count = 0;
        int8_t capacity = 0;
    } pattern_lines[5];
    
    Tile floor_line[7];
    int8_t floor_count;
    bool round_just_finished;

    PlayerBoard() : score(0), floor_count(0), round_just_finished(false) {
        std::memset(wall, EMPTY, sizeof(wall));
        for(int i=0; i<5; ++i) {
            pattern_lines[i].capacity = i + 1;
            pattern_lines[i].count = 0;
            pattern_lines[i].color = EMPTY;
        }
    }

    static int8_t get_wall_color(int8_t row, int8_t col) {
        return (col - row + 5) % 5;
    }

    bool can_place_in_pattern_line(int8_t row_idx, int8_t color) const {
        if (row_idx == -1) return true; // 地板永远可以放
        
        // 防御性检查：确保不是 FIRST_PLAYER 标记进入了这里
        if (color < 0 || color > 4) return false; 

        const auto& line = pattern_lines[row_idx];
        
        // 1. 如果该行已经满了，不能放
        if (line.count >= line.capacity) return false;
        
        // 2. 如果该行已有瓷砖，且颜色不匹配，不能放
        if (line.color != EMPTY && line.color != color) return false;
        
        // 3. 检查墙上是否已有该颜色 (这是你反馈的 Bug 核心)
        // 确保这里的计算逻辑与 get_wall_color 以及前端渲染完全同步
        int8_t target_col = (color + row_idx) % 5;
        if (wall[row_idx][target_col] != EMPTY) return false;
        
        return true;
    }

    void add_tiles(int8_t row_idx, int8_t color, int8_t quantity) {
        int8_t remaining = quantity;
        if (row_idx != -1) {
            auto& line = pattern_lines[row_idx];
            if (line.count == 0) line.color = color;
            int8_t space = line.capacity - line.count;
            int8_t take = std::min(space, remaining);
            line.count += take;
            remaining -= take;
        }
        for (int i = 0; i < remaining && floor_count < 7; ++i) {
            floor_line[floor_count++] = color;
        }
    }

    bool score_round(std::vector<Tile>& discard_pile) {
        bool game_ending = false;
        round_just_finished = true;
        for (int r = 0; r < 5; ++r) {
            auto& line = pattern_lines[r];
            if (line.count == line.capacity) {
                int8_t c_idx = (line.color + r) % 5;
                wall[r][c_idx] = line.color;
                score += calculate_gain(r, c_idx);
                for (int i = 0; i < line.capacity - 1; ++i) discard_pile.push_back(line.color);
                line.count = 0; line.color = EMPTY;
            }
        }
        int32_t penalty = 0;
        for (int i = 0; i < floor_count; ++i) penalty += FLOOR_PENALTIES[i];
        score = std::max(0, score + penalty);
        for (int i=0; i<floor_count; ++i) if (floor_line[i] != FIRST_PLAYER) discard_pile.push_back(floor_line[i]);
        floor_count = 0;
        for (int r = 0; r < 5; ++r) {
            bool full = true;
            for (int c = 0; c < 5; ++c) if (wall[r][c] == EMPTY) { full = false; break; }
            if (full) game_ending = true;
        }
        return game_ending;
    }

    void end_game_bonus() {
        for (int r = 0; r < 5; ++r) {
            bool full = true;
            for (int c = 0; c < 5; ++c) if (wall[r][c] == EMPTY) full = false;
            if (full) score += 2;
        }
        for (int c = 0; c < 5; ++c) {
            bool full = true;
            for (int r = 0; r < 5; ++r) if (wall[r][c] == EMPTY) full = false;
            if (full) score += 7;
        }
        for (int color = 0; color < 5; ++color) {
            int cnt = 0;
            for (int r = 0; r < 5; ++r) for (int c = 0; c < 5; ++c) if (wall[r][c] == color) cnt++;
            if (cnt == 5) score += 10;
        }
    }

private:
    int32_t calculate_gain(int r, int c) {
        int h = 0, v = 0;
        for (int i = c - 1; i >= 0 && wall[r][i] != EMPTY; --i) h++;
        for (int i = c + 1; i < 5 && wall[r][i] != EMPTY; ++i) h++;
        for (int i = r - 1; i >= 0 && wall[i][c] != EMPTY; --i) v++;
        for (int i = r + 1; i < 5 && wall[i][c] != EMPTY; ++i) v++;
        int res = (h > 0 ? h + 1 : 0) + (v > 0 ? v + 1 : 0);
        return res == 0 ? 1 : res;
    }
};

class AzulGame {
public:
    int8_t num_players, num_factories, current_player_idx, round_number;
    PlayerBoard players[2]; 
    std::vector<Tile> bag, discard_pile;
    Tile factories[9][4]; 
    Tile center[MAX_CENTER];
    int8_t center_count;
    int8_t first_player_token_owner;
    bool game_over;

    AzulGame(int n = 2, unsigned int seed = 42) : num_players(n), num_factories(n*2+1), current_player_idx(0), round_number(0), center_count(0), first_player_token_owner(-1), game_over(false) {
        for (Tile c = 0; c < 5; ++c) for (int i = 0; i < 20; ++i) bag.push_back(c);
        // unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::shuffle(bag.begin(), bag.end(), std::mt19937(seed));
        start_round();
    }

    int8_t get_tile_count(int8_t source_id, int8_t color) const {
        int8_t count = 0;
        if (source_id == -1) {
            for (int i = 0; i < center_count; ++i) if (center[i] == color) count++;
        } else {
            for (int i = 0; i < 4; ++i) if (factories[source_id][i] == color) count++;
        }
        return count;
    }

    void start_round() {
        round_number++;
        for (int i = 0; i < num_factories; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (bag.empty()) {
                    bag = discard_pile; discard_pile.clear();
                    std::shuffle(bag.begin(), bag.end(), std::mt19937(std::random_device{}()));
                }
                factories[i][j] = bag.empty() ? EMPTY : bag.back();
                if (!bag.empty()) bag.pop_back();
            }
        }
        center_count = 0;
        center[center_count++] = FIRST_PLAYER;
        if (first_player_token_owner != -1) current_player_idx = first_player_token_owner;
    }

    std::vector<Action> get_legal_actions() const {
        std::vector<Action> actions;
        actions.reserve(32);
        const auto& p = players[current_player_idx];
        
        auto check = [&](int8_t src, int8_t color) {
            for (int8_t r = 0; r < 5; ++r) if (p.can_place_in_pattern_line(r, color)) actions.push_back({src, color, r});
            actions.push_back({src, color, -1});
        };

        for (int8_t i = 0; i < num_factories; ++i) {
            bool seen[5] = {false};
            for (int j = 0; j < 4; ++j) {
                Tile c = factories[i][j];
                if (c >= 0 && !seen[c]) { seen[c] = true; check(i, c); }
            }
        }
        bool c_seen[5] = {false};
        for (int i = 0; i < center_count; ++i) {
            Tile c = center[i];
            if (c >= 0 && c < 5 && !c_seen[c]) { c_seen[c] = true; check(-1, c); }
        }
        return actions;
    }

    MoveRecord step(const Action& action, bool is_sim = false) {
        MoveRecord rec;
        rec.action = action; rec.player_idx = current_player_idx;
        rec.first_player_token_owner_prev = first_player_token_owner;
        rec.round_ended = false; rec.first_player_taken = false;

        auto& p = players[current_player_idx];
        p.round_just_finished = false;
        rec.prev_score = p.score;
        std::memcpy(rec.prev_floor_line, p.floor_line, 7);
        rec.prev_floor_count = p.floor_count;
        if (action.target_row != -1) {
            rec.prev_line_count = p.pattern_lines[action.target_row].count;
            rec.prev_line_color = p.pattern_lines[action.target_row].color;
        }

        int8_t taken_count = 0;
        if (action.source_id == -1) {
            int8_t write_idx = 0;
            for (int i = 0; i < center_count; ++i) {
                if (center[i] == FIRST_PLAYER) {
                    if (p.floor_count < 7) p.floor_line[p.floor_count++] = FIRST_PLAYER;
                    first_player_token_owner = current_player_idx;
                    rec.first_player_taken = true;
                } else if (center[i] == action.color) {
                    taken_count++;
                } else {
                    center[write_idx++] = center[i];
                }
            }
            center_count = write_idx;
        } else {
            std::memcpy(rec.factory_prev_state, factories[action.source_id], 4);
            for (int i = 0; i < 4; ++i) {
                Tile c = factories[action.source_id][i];
                if (c == action.color) taken_count++;
                else if (c != EMPTY) center[center_count++] = c;
            }
            std::memset(factories[action.source_id], EMPTY, 4);
        }

        p.add_tiles(action.target_row, action.color, taken_count);

        bool round_done = (center_count == 0);
        if (round_done) {
            for (int i=0; i<num_factories; ++i) {
                for (int j=0; j<4; ++j) if(factories[i][j] != EMPTY) round_done = false;
            }
        }

        if (round_done) {
            rec.round_ended = true;
            bool end = false;
            for (auto& pl : players) if (pl.score_round(discard_pile)) end = true;
            if (end) { 
                game_over = true; 
                for (auto& pl : players) pl.end_game_bonus(); 
            } else {
                start_round();
            }
        } else {
            current_player_idx = (current_player_idx + 1) % num_players;
        }
        return rec;
    }

    // 关键：导出 JSON 供前端使用
    std::string to_json() const {
        std::stringstream ss;
        ss << "{";
        ss << "\"round\":" << (int)round_number << ",";
        ss << "\"currentPlayer\":" << (int)current_player_idx << ",";
        ss << "\"gameOver\":" << (game_over ? "true" : "false") << ",";
        
        ss << "\"factories\":[";
        for(int i=0; i<num_factories; ++i) {
            ss << "[";
            for(int j=0; j<4; ++j) ss << (int)factories[i][j] << (j==3?"":",");
            ss << "]" << (i==num_factories-1?"":",");
        }
        ss << "],";

        ss << "\"center\":[";
        for(int i=0; i<center_count; ++i) ss << (int)center[i] << (i==center_count-1?"":",");
        ss << "],";

        ss << "\"players\":[";
        for(int p=0; p<2; ++p) {
            const auto& pb = players[p];
            ss << "{";
            ss << "\"score\":" << pb.score << ",";
            ss << "\"wall\":[";
            for(int r=0; r<5; ++r) {
                ss << "[";
                for(int c=0; c<5; ++c) ss << (int)pb.wall[r][c] << (c==4?"":",");
                ss << "]" << (r==4?"":",");
            }
            ss << "],";
            ss << "\"patterns\":[";
            for(int r=0; r<5; ++r) {
                ss << "{\"color\":" << (int)pb.pattern_lines[r].color << ",\"count\":" << (int)pb.pattern_lines[r].count << ",\"cap\":" << (int)pb.pattern_lines[r].capacity << "}" << (r==4?"":",");
            }
            ss << "],";
            ss << "\"floor\":[";
            for(int i=0; i<pb.floor_count; ++i) ss << (int)pb.floor_line[i] << (i==pb.floor_count-1?"":",");
            ss << "]";
            ss << "}" << (p==1?"":",");
        }
        ss << "]";
        ss << "}";
        return ss.str();
    }
};

#endif