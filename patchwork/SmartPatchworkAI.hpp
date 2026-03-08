#pragma once
#include "patchworkGame.hpp"
#include "patch_data.h"
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
        children.reserve(24); 
    }
    ~SmartMCTSNode() { for (auto c : children) delete c; }
};

class SmartPatchworkAI {
    double time_limit;
    int max_depth;

    std::string action_to_string(const Action& a, const std::vector<PieceData>& pieces) {
        if (a.type == 0) return "前进 (Move Forward)";
        if (a.type == 2) return "放置皮革 (1x1 Leather)";
        const auto& p = pieces[a.piece_id];
        return "购买 ID:" + std::to_string(p.id) + " [纽扣:" + std::to_string(p.cost) + "]";
    }

    // 启发式权重：引导搜索方向
    double get_action_weight(const PatchworkGame& game, const Action& action, const std::vector<PieceData>& pieces) {
        if (action.type == 2) return 100.0; // 补丁必拿，价值极高
        if (action.type == 0) return 2.0;   // 前进是最后的保底

        const auto& p = pieces[action.piece_id];
        double weight = 10.0;
        
        // 核心逻辑：早期重收入，后期重纽扣结余
        double time_factor = (53.0 - game.players[game.current_player_idx].time_pos) / 53.0;
        weight += p.income * (10.0 * time_factor); 
        weight += (p.cost < game.players[game.current_player_idx].buttons) ? 5.0 : -10.0;
        
        return std::max(0.1, weight);
    }

    // --- 核心评估函数：包含了 7x7 奖赏逻辑 ---
    double evaluate_potential(const PatchworkGame& game, int p_idx) {
        const auto& p = game.players[p_idx];
        double score = (double)p.buttons;
        
        int filled_count = p.count_filled();
        // 1. 空位惩罚 (每空一格 -2 分)
        score -= (81 - filled_count) * 2.0;

        // 2. 7x7 奖赏逻辑
        if (p.has_7x7_bonus) {
            score += 7.0; // 已获得奖赏
        } else if (!game.bonus_claimed) {
            // 启发式：如果还没人领奖，且填充数接近，给予潜能分鼓励竞争
            if (filled_count >= 30) {
                score += (filled_count - 30) * 0.2; 
            }
        }

        // 3. 纽扣收入预估 (随时间减少)
        double remaining_time = 53.0 - p.time_pos;
        // 粗略估计还能触发几次收入结算 (每 9 格一次)
        double future_income_triggers = std::max(0.0, remaining_time / 9.0);
        score += p.income * future_income_triggers;

        return score;
    }

    std::vector<std::pair<Action, double>> get_prior_actions(const PatchworkGame& game, const std::vector<PieceData>& pieces) {
        auto actions = game.get_legal_actions(pieces);
        std::vector<std::pair<Action, double>> res;
        double sum = 0.0;
        for(const auto& a : actions) sum += get_action_weight(game, a, pieces);
        for(const auto& a : actions) {
            res.push_back(std::make_pair(a, get_action_weight(game, a, pieces) / (sum + 1e-9)));
        }
        return res;
    }

    SmartMCTSNode* run_mcts_worker(PatchworkGame real_game, const std::vector<PieceData>& pieces, double limit, int seed) {
        std::mt19937 local_rng(seed);
        SmartMCTSNode* root = new SmartMCTSNode(nullptr, {0, -1, 0}, real_game.current_player_idx, 1.0);
        root->untried_actions = get_prior_actions(real_game, pieces);
        
        auto start_time = std::chrono::steady_clock::now();
        int sim_count = 0;

        while (true) {
            if ((sim_count & 63) == 0) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (std::chrono::duration<double>(elapsed).count() >= limit) break;
            }

            PatchworkGame sim_game = real_game;
            SmartMCTSNode* node = root;

            // Selection (PUCT)
            while (node->untried_actions.empty() && !node->children.empty() && !sim_game.game_over) {
                bool is_me = (sim_game.current_player_idx == root->player_idx);
                double best_s = is_me ? -1e18 : 1e18;
                SmartMCTSNode* best_c = nullptr;

                for (auto* child : node->children) {
                    double q = (child->visits > 0) ? (child->value / child->visits) : 0.0;
                    double u = 1.41 * child->prior_prob * std::sqrt(node->visits) / (1 + child->visits);
                    double score = is_me ? (q + u) : (q - u);
                    if (is_me ? (score > best_s) : (score < best_s)) { best_s = score; best_c = child; }
                }
                if (!best_c) break;
                node = best_c;
                sim_game.apply_action(node->action, pieces);
            }

            // Expansion
            if (!node->untried_actions.empty() && !sim_game.game_over) {
                std::uniform_int_distribution<> d(0, (int)node->untried_actions.size() - 1);
                int idx = d(local_rng);
                auto act_pair = node->untried_actions[idx];
                node->untried_actions.erase(node->untried_actions.begin() + idx);
                
                sim_game.apply_action(act_pair.first, pieces);
                SmartMCTSNode* child = new SmartMCTSNode(node, act_pair.first, sim_game.current_player_idx, act_pair.second);
                child->untried_actions = get_prior_actions(sim_game, pieces);
                node->children.push_back(child);
                node = child;
            }

            // Rollout
            int depth = 0;
            while (!sim_game.game_over && depth < max_depth) {
                auto acts = sim_game.get_legal_actions(pieces);
                if (acts.empty()) break;
                // 快速选择动作
                std::uniform_int_distribution<> d(0, (int)acts.size() - 1);
                sim_game.apply_action(acts[d(local_rng)], pieces);
                depth++;
            }

            // Backprop
            double my_e = evaluate_potential(sim_game, root->player_idx);
            double op_e = evaluate_potential(sim_game, 1 - root->player_idx);
            double reward = std::tanh((my_e - op_e) / 30.0);

            SmartMCTSNode* back_node = node;
            while (back_node) {
                back_node->visits++;
                back_node->value += reward;
                back_node = back_node->parent;
            }
            sim_count++;
        }
        return root;
    }

public:
    SmartPatchworkAI(double t = 1.0, int d = 40) : time_limit(t), max_depth(d) {}

    Action get_best_action(PatchworkGame real_game, const std::vector<PieceData>& pieces) {
        auto start_t = std::chrono::steady_clock::now();
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 2;
        
        std::vector<std::future<SmartMCTSNode*>> futures;
        for (unsigned int i = 0; i < n_threads; ++i) {
            futures.push_back(std::async(std::launch::async, &SmartPatchworkAI::run_mcts_worker, this, real_game, std::ref(pieces), time_limit, (int)std::random_device{}()));
        }

        struct ActionStats { int v = 0; double val = 0; };
        std::map<uint128, ActionStats> stats_map;
        std::vector<Action> actions_list;

        for (auto& f : futures) {
            SmartMCTSNode* t_root = f.get();
            for (auto* child : t_root->children) {
                // 动作唯一性识别：由类型、块ID和位掩码共同决定
                uint128 key = child->action.mask ^ ((uint128)child->action.type << 120) ^ ((uint128)child->action.piece_id << 110);
                if (stats_map.find(key) == stats_map.end()) actions_list.push_back(child->action);
                stats_map[key].v += child->visits;
                stats_map[key].val += child->value;
            }
            delete t_root;
        }

        std::sort(actions_list.begin(), actions_list.end(), [&](const Action& a, const Action& b){
            uint128 key_a = a.mask ^ ((uint128)a.type << 120) ^ ((uint128)a.piece_id << 110);
            uint128 key_b = b.mask ^ ((uint128)b.type << 120) ^ ((uint128)b.piece_id << 110);
            return stats_map[key_a].v > stats_map[key_b].v;
        });

        // 绚丽的控制台报告
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_t).count();
        std::cout << "\033[36m[AI 思考完成] 耗时:" << std::fixed << std::setprecision(2) << elapsed << "s | 方案数:" << actions_list.size() << "\033[0m" << std::endl;

        return actions_list[0];
    }
};