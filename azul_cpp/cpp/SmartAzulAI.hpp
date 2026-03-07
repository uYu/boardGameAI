#ifndef SMART_AZUL_AI_HPP
#define SMART_AZUL_AI_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <string>
#include "AzulGame.hpp"

// MCTS 节点结构
struct SmartMCTSNode {
    Action action;
    SmartMCTSNode* parent;
    std::vector<SmartMCTSNode*> children;
    
    // 算法核心数据
    int visits = 0;
    double value = 0.0;
    double prior_prob = 1.0; 
    std::vector<std::pair<Action, double>> untried_actions;
    int player_idx;

    SmartMCTSNode(Action a, SmartMCTSNode* p = nullptr, int idx = 0, double prior = 1.0) 
        : action(a), parent(p), player_idx(idx), prior_prob(prior) {
        children.reserve(32);
    }

    ~SmartMCTSNode() {
        for (auto* child : children) delete child;
    }

    // 使用 PUCT 公式选择子节点 (结合先验概率)
    SmartMCTSNode* select_child(bool is_ai) {
        double best_s = is_ai ? -1e18 : 1e18;
        SmartMCTSNode* best_c = nullptr;

        for (auto* child : children) {
            double q = (child->visits > 0) ? (child->value / child->visits) : 0.0;
            // CPUCT 常数设为 2.0，平衡探索与先验知识
            double u = 2.0 * child->prior_prob * std::sqrt((double)this->visits) / (1.0 + child->visits);
            double score = is_ai ? (q + u) : (q - u);

            if (is_ai ? (score > best_s) : (score < best_s)) {
                best_s = score;
                best_c = child;
            }
        }
        return best_c;
    }

    void update(double res) {
        visits++;
        value += res;
    }
};

struct AIDecision {
    Action best_action;
    double win_rate = 0.0;
    int total_sims = 0;
    double elapsed_time = 0.0;
};

class SmartAzulAI {
public:
    double time_limit;
    int max_depth = 40; // 防止模拟陷入死循环

    SmartAzulAI(double limit) : time_limit(limit) {}

    AIDecision get_best_decision(const AzulGame& real_game) {
        auto start_t = std::chrono::steady_clock::now();
        
        // 运行修复后的核心逻辑
        SmartMCTSNode* root = this->run_mcts_worker(real_game, time_limit * 0.95);

        AIDecision decision;
        decision.total_sims = root->visits;
        decision.elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_t).count();

        if (!root->children.empty()) {
            // 最终决策：选择访问次数最多的，这比选胜率最高的更稳健
            auto best_it = std::max_element(root->children.begin(), root->children.end(),
                [](SmartMCTSNode* a, SmartMCTSNode* b) { return a->visits < b->visits; });
            
            decision.best_action = (*best_it)->action;
            decision.win_rate = ((*best_it)->value / (*best_it)->visits + 1.0) / 2.0;
        }

        delete root; 
        return decision;
    }

private:
    // --- 算法策略修复 1: 动作权重评价 (移植自老代码) ---
    double get_action_weight(const AzulGame& game, const Action& action) {
        if (action.target_row == -1) return 1.0; // 地板位基础权重
        double weight = 10.0; 
        
        const auto& player = game.players[game.current_player_idx];
        const auto& line = player.pattern_lines[action.target_row];
        int needed = line.capacity - line.count;
        int available = game.get_tile_count(action.source_id, action.color);
        
        weight += available * 5.0; 
        if (available >= 3 && needed >= available) weight += 40.0; 
        
        int target_col = (action.color + action.target_row) % 5;
        if (target_col == 2) weight += 25.0; // 优先中间列
        else if (target_col == 1 || target_col == 3) weight += 10.0;
        
        if (available > needed) weight -= (available - needed) * 8.0; // 溢出惩罚
        
        return std::max(0.1, weight); 
    }

    // --- 算法策略修复 2: 局势评估打分 (移植自老代码) ---
    double evaluate_potential(const PlayerBoard& p) {
        double pot = (double)p.score;
        for (int r = 0; r < 5; r++) {
            if (p.pattern_lines[r].count > 0) pot += 0.5 * p.pattern_lines[r].count;
        }
        for (int c = 0; c < 5; c++) {
            int r_cnt = 0;
            for (int r = 0; r < 5; r++) if (p.wall[r][c] != EMPTY) r_cnt++;
            if (r_cnt == 4) pot += 5.0; else if (r_cnt == 3) pot += 2.0;
        }
        int penalty = 0;
        for (int i = 0; i < (int)p.floor_count; ++i) {
            penalty += (i < 7) ? FLOOR_PENALTIES[i] : -3;
        }
        return pot + (double)penalty;
    }

    // 获取带权重的合法动作列表
    std::vector<std::pair<Action, double>> get_prior_actions(const AzulGame& game) {
        auto actions = game.get_legal_actions();
        std::vector<std::pair<Action, double>> res;
        double sum = 0.0;
        for(const auto& a : actions) sum += get_action_weight(game, a);
        for(const auto& a : actions) {
            res.push_back({a, get_action_weight(game, a) / (sum + 1e-9)});
        }
        return res;
    }

    SmartMCTSNode* run_mcts_worker(const AzulGame& game, double limit) {
        unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
        std::mt19937 gen(seed);
        
        SmartMCTSNode* root = new SmartMCTSNode(Action{-1, -1, -1}, nullptr, game.current_player_idx);
        root->untried_actions = get_prior_actions(game);

        auto start_time = std::chrono::steady_clock::now();
        int iterations = 0;

        while (true) {
            if ((iterations & 63) == 0) {
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() >= limit) break;
            }

            SmartMCTSNode* node = root;
            AzulGame sim_game = game;

            // 1. Selection (使用修复后的 PUCT)
            while (node->untried_actions.empty() && !node->children.empty() && !sim_game.game_over) {
                bool is_ai = (sim_game.current_player_idx == root->player_idx);
                node = node->select_child(is_ai);
                sim_game.step(node->action, true);
            }

            // 2. Expansion (带有先验概率)
            if (!node->untried_actions.empty() && !sim_game.game_over) {
                std::uniform_int_distribution<> d(0, (int)node->untried_actions.size() - 1);
                int idx = d(gen);
                auto act_pair = node->untried_actions[idx];
                node->untried_actions.erase(node->untried_actions.begin() + idx);

                sim_game.step(act_pair.first, true);
                SmartMCTSNode* child = new SmartMCTSNode(act_pair.first, node, sim_game.current_player_idx, act_pair.second);
                child->untried_actions = get_prior_actions(sim_game);
                node->children.push_back(child);
                node = child;
            }

            // 3. Simulation (带有权重的快速 Rollout)
            int d = 0;
            while (!sim_game.game_over && d < max_depth) {
                auto acts = sim_game.get_legal_actions();
                if (acts.empty()) break;
                
                double tw = 0;
                for (const auto& a : acts) tw += get_action_weight(sim_game, a);
                double target = std::uniform_real_distribution<double>(0, tw)(gen);
                
                int chosen = 0;
                for (size_t i = 0; i < acts.size(); ++i) {
                    target -= get_action_weight(sim_game, acts[i]);
                    if (target <= 0) { chosen = (int)i; break; }
                }
                sim_game.step(acts[chosen], true);
                d++;
            }

            // 4. Backpropagation (使用启发式分值评估)
            double my_e = evaluate_potential(sim_game.players[root->player_idx]);
            double opp_e = 0;
            for(int i = 0; i < sim_game.num_players; ++i) {
                if(i != root->player_idx) opp_e += evaluate_potential(sim_game.players[i]);
            }
            // 将分差映射到 [-1, 1] 区间
            double reward = std::max(-1.0, std::min(1.0, (my_e - (opp_e / (sim_game.num_players - 1))) / 20.0));
            
            SmartMCTSNode* back_node = node;
            while (back_node) {
                back_node->update(reward);
                back_node = back_node->parent;
            }
            iterations++;
        }
        return root;
    }
};

#endif