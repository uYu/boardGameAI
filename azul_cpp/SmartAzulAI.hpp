#pragma once
#include "AzulGame.hpp"
#include <cmath>
#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <string>
#include <iomanip>
#include <future>
#include <vector>
#include <map>
#include <thread>

class SmartMCTSNode {
public:
    SmartMCTSNode* parent;
    Action action;
    std::vector<SmartMCTSNode*> children;
    int visits = 0;
    double value = 0.0;
    double prior_prob = 1.0; 
    std::vector<std::pair<Action, double>> untried_actions;
    int player_idx;

    SmartMCTSNode(SmartMCTSNode* p, Action a, int idx, double prior) 
        : parent(p), action(a), player_idx(idx), prior_prob(prior) {
        children.reserve(32); 
    }
    ~SmartMCTSNode() { for (auto c : children) delete c; }
};

class SmartAzulAI {
    double time_limit;
    int max_depth;

    struct ActionCompare {
        bool operator()(const Action& a, const Action& b) const {
            if (a.source_id != b.source_id) return a.source_id < b.source_id;
            if (a.color != b.color) return a.color < b.color;
            return a.target_row < b.target_row;
        }
    };

    std::string action_to_string(const Action& a) {
        if (a.source_id == -2) return "ROOT";
        std::string src = (a.source_id == -1) ? "Center" : "F" + std::to_string(a.source_id + 1);
        std::string col;
        switch(a.color) {
            case 0: col = "BLUE  "; break;
            case 1: col = "YELLOW"; break;
            case 2: col = "RED   "; break;
            case 3: col = "BLACK "; break;
            case 4: col = "WHITE "; break;
            default: col = "UNKNOWN"; break;
        }
        std::string row = (a.target_row == -1) ? "Floor" : "Row " + std::to_string(a.target_row + 1);
        return src + " -> [" + col + "] -> " + row;
    }

    double get_action_weight(const AzulGame& game, const Action& action) {
        if (action.target_row == -1) return 1.0;
        double weight = 10.0; 
        if (game.round_number == 1) {
            const auto& player = game.players[game.current_player_idx];
            const auto& line = player.pattern_lines[action.target_row];
            int needed = line.capacity - line.count;
            int available = game.get_tile_count(action.source_id, action.color);
            
            weight += available * 5.0; 
            if (available >= 3 && needed >= available) weight += 40.0; 
            int target_col = (action.color + action.target_row) % 5;
            if (target_col == 2) weight += 25.0; 
            else if (target_col == 1 || target_col == 3) weight += 10.0;
            if (available > needed) weight -= (available - needed) * 8.0; 
        }
        return std::max(0.1, weight); 
    }

    std::vector<std::pair<Action, double>> get_prior_actions(const AzulGame& game) {
        auto actions = game.get_legal_actions();
        std::vector<std::pair<Action, double>> res;
        double sum = 0.0;
        for(const auto& a : actions) sum += get_action_weight(game, a);
        for(const auto& a : actions) {
            res.push_back(std::make_pair(a, get_action_weight(game, a) / (sum + 1e-9)));
        }
        return res;
    }

    double evaluate_potential(const PlayerBoard& p) {
        double pot = (double)p.score;

        // 1. 评估 Pattern Lines 的潜在价值
        // 原逻辑：每有一块砖加 0.5 分，鼓励填充
        for (int r = 0; r < 5; r++) {
            if (p.pattern_lines[r].count > 0) {
                pot += 0.5 * p.pattern_lines[r].count;
            }
        }

        // 2. 墙面列奖励预测 (↕️ 列)
        // 检查每一列，如果快满了（3或4块），给予额外权重
        for (int c = 0; c < 5; c++) {
            int r_cnt = 0;
            for (int r = 0; r < 5; r++) {
                if (p.wall[r][c] != EMPTY) r_cnt++;
            }
            if (r_cnt == 4) pot += 5.0; 
            else if (r_cnt == 3) pot += 2.0;
        }

        // 3. 颜色奖励预测 (🎨 颜色)
        // 检查每种颜色，如果快凑齐 5 个了，给予额外权重
        for (int color = 0; color < 5; color++) {
            int cnt = 0;
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 5; c++) {
                    if (p.wall[r][c] == color) cnt++;
                }
            }
            if (cnt == 4) pot += 7.0; 
            else if (cnt == 3) pot += 3.0;
        }

        // 4. 地板扣分计算 (💀 地板)
        // 修复点：改用 p.floor_count 遍历，并硬编码索引上限 7
        int penalty = 0;
        for (int i = 0; i < (int)p.floor_count; ++i) {
            // FLOOR_PENALTIES 是我们在 AzulGame.hpp 定义的固定 7 长度数组
            penalty += (i < 7) ? FLOOR_PENALTIES[i] : -3;
        }

        return pot + (double)penalty;
    }

    SmartMCTSNode* run_mcts_worker(AzulGame real_game, double limit, int seed) {
    std::mt19937 local_rng(seed);
    SmartMCTSNode* root = new SmartMCTSNode(nullptr, {-2, -2, -2}, real_game.current_player_idx, 1.0);
    root->untried_actions = get_prior_actions(real_game);
    
    auto start_time = std::chrono::steady_clock::now();
    int sim_count = 0;
    AzulGame sim_game = real_game; 
    std::vector<MoveRecord> history;
    history.reserve(128);

    while (true) {
        if ((sim_count & 255) == 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration<double>(elapsed).count() >= limit) break;
        }

        SmartMCTSNode* node = root;
        history.clear();
        bool cross_round_happened = false;

        // Selection
        while (node->untried_actions.empty() && !node->children.empty() && !sim_game.game_over) {
            bool is_ai = (sim_game.current_player_idx == root->player_idx);
            double best_s = is_ai ? -1e18 : 1e18;
            SmartMCTSNode* best_c = nullptr;
            for (auto* child : node->children) {
                double q = (child->visits > 0) ? (child->value / child->visits) : 0.0;
                double u = 2.0 * child->prior_prob * std::sqrt(node->visits) / (1 + child->visits);
                double score = is_ai ? (q + u) : (q - u);
                if (is_ai ? (score > best_s) : (score < best_s)) { best_s = score; best_c = child; }
            }
            if (!best_c) break;
            node = best_c;
            
            // --- 修改点：增加 true 标志 ---
            MoveRecord rec = sim_game.step(node->action, true); 
            history.push_back(rec);
            // if (rec.round_ended) { cross_round_happened = true; break; }
        }

        // Expansion
        if (!node->untried_actions.empty() && !sim_game.game_over && !cross_round_happened) {
            std::uniform_int_distribution<> d(0, (int)node->untried_actions.size() - 1);
            int idx = d(local_rng);
            auto act_pair = node->untried_actions[idx];
            node->untried_actions.erase(node->untried_actions.begin() + idx);
            
            // --- 修改点：增加 true 标志 ---
            MoveRecord rec = sim_game.step(act_pair.first, true); 
            history.push_back(rec);
            
            SmartMCTSNode* child = new SmartMCTSNode(node, act_pair.first, sim_game.current_player_idx, act_pair.second);
            child->untried_actions = get_prior_actions(sim_game);
            node->children.push_back(child);
            node = child;
            if (rec.round_ended) cross_round_happened = true;
        }

        // Rollout
        int depth = 0;
        while (!sim_game.game_over && depth < max_depth && !cross_round_happened) {
            auto acts = sim_game.get_legal_actions();
            if (acts.empty()) break;
            double tw = 0;
            for (const auto& a : acts) tw += get_action_weight(sim_game, a);
            double target = std::uniform_real_distribution<double>(0, tw)(local_rng);
            int chosen = 0;
            for (size_t i = 0; i < acts.size(); ++i) {
                target -= get_action_weight(sim_game, acts[i]);
                if (target <= 0) { chosen = (int)i; break; }
            }
            
            // --- 修改点：增加 true 标志 ---
            MoveRecord rec = sim_game.step(acts[chosen], true); 
            history.push_back(rec);
            if (rec.round_ended) cross_round_happened = true;
            depth++;
        }

        // Backprop
        double my_e = evaluate_potential(sim_game.players[root->player_idx]);
        double opp_e = 0;
        for(int i = 0; i < sim_game.num_players; ++i) if(i != root->player_idx) opp_e += evaluate_potential(sim_game.players[i]);
        double reward = std::max(-1.0, std::min(1.0, (my_e - (opp_e / (sim_game.num_players - 1))) / 20.0));

        SmartMCTSNode* back_node = node;
        while (back_node) {
            back_node->visits++;
            back_node->value += reward;
            back_node = back_node->parent;
        }

        // Undo
        sim_game = real_game;
        // if (cross_round_happened) {
        //     sim_game = real_game;
        // } else {
        //     for (int i = (int)history.size() - 1; i >= 0; --i) sim_game.undo_step(history[i]);
        // }
        sim_count++;
    }
    return root;
}

public:
    struct ActionStats { int visits; double value; double prior; };
    SmartAzulAI(double t = 1.0, int d = 50) : time_limit(t), max_depth(d) {}

    Action get_best_action(AzulGame real_game) {
        auto start_t = std::chrono::steady_clock::now();
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 2;
        
        std::vector<std::future<SmartMCTSNode*>> futures;
        for (unsigned int i = 0; i < n_threads; ++i) {
            futures.push_back(std::async(std::launch::async, &SmartAzulAI::run_mcts_worker, this, real_game, time_limit, (int)std::random_device{}()));
        }

        std::map<Action, ActionStats, ActionCompare> combined;
        int total_sims = 0;
        SmartMCTSNode* best_thread_root = nullptr;

        for (auto& f : futures) {
            SmartMCTSNode* t_root = f.get();
            total_sims += t_root->visits;
            for (auto* child : t_root->children) {
                ActionStats& s = combined[child->action];
                s.visits += child->visits;
                s.value += child->value;
                s.prior = child->prior_prob;
            }
            if (!best_thread_root || t_root->visits > best_thread_root->visits) {
                if(best_thread_root) delete best_thread_root;
                best_thread_root = t_root;
            } else {
                delete t_root;
            }
        }

        // --- 绚丽的 DEBUG 报告 ---
        auto end_t = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end_t - start_t).count();
        std::cout << "\n\033[36m[高性能 MCTS 引擎报告] 耗时: " << std::fixed << std::setprecision(2) << elapsed << "s\033[0m\n";
        
        double my_p = evaluate_potential(real_game.players[real_game.current_player_idx]);
        double op_p = evaluate_potential(real_game.players[1 - real_game.current_player_idx]);
        std::cout << " ├─ 局势评估: AI " << (my_p >= op_p ? "领先 " : "落后 ") << std::abs(my_p - op_p) << " 分\n";
        std::cout << " ├─ 算力统计: 线程数 " << n_threads << " | 总模拟 " << total_sims << " 次\n";

        // PV 路线 (Principal Variation)
        std::cout << " ├─ 预测路线 (PV):\n";
        SmartMCTSNode* curr = best_thread_root;
        int step_count = 1;
        while (curr && !curr->children.empty() && step_count <= 4) {
            SmartMCTSNode* bc = nullptr;
            for (auto* c : curr->children) if (!bc || c->visits > bc->visits) bc = c;
            if (!bc || bc->visits < 5) break;
            std::string p_tag = (curr->player_idx == real_game.current_player_idx) ? "\033[36mAI \033[0m" : "\033[35mYou\033[0m";
            std::cout << " │    Step " << step_count << " (" << p_tag << "): " << action_to_string(bc->action) << "\n";
            curr = bc; step_count++;
        }

        // Top 候选动作
        std::cout << " └─ Top 候选动作:\n";
        std::vector<std::pair<Action, ActionStats>> sorted;
        for (auto const& kv : combined) sorted.push_back(kv);
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return a.second.visits > b.second.visits; });

        for (int i = 0; i < std::min((int)sorted.size(), 5); ++i) {
            auto& res = sorted[i];
            double win_rate = (res.second.value / res.second.visits + 1.0) / 2.0;
            std::cout << "    " << (i+1) << ". " << action_to_string(res.first) 
                      << " | 模拟: " << std::setw(7) << res.second.visits 
                      << " | 胜率: " << std::fixed << std::setprecision(1) << (win_rate * 100.0) << "%\n";
        }

        Action final_act = sorted[0].first;
        std::cout << "\n\033[38;5;220m[AI 最终决定] " << action_to_string(final_act) << "\033[0m\n";
        
        if (best_thread_root) delete best_thread_root;
        std::cout << "\033[2m(按回车键继续...)\033[0m";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return final_act;
    }
};