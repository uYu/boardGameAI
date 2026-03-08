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
#include <thread>

// 节点结构：保持精简
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

class SmartPatchworkAI {
    double time_limit;
    int max_depth;

    // 启发式权重：引导搜索方向
    double get_action_weight(const PatchworkGame& game, const Action& action) {
        if (action.type == 2) return 150.0; // 补丁必拿，价值极高
        if (action.type == 0) return 1.5;   // 前进是保底，权重稍低

        const auto& p = ALL_PIECES[action.piece_id];
        double weight = 10.0;
        
        // 核心逻辑：早期重收入，后期重纽扣结余
        double progress = game.players[game.current_player_idx].time_pos / 53.0;
        weight += p.income * (15.0 * (1.0 - progress)); 
        weight -= p.cost * 0.5; // 稍微考虑成本
        
        return std::max(0.1, weight);
    }

    // --- 核心评估函数 ---
    double evaluate_potential(const PatchworkGame& game, int p_idx) {
        const auto& p = game.players[p_idx];
        double score = (double)p.buttons;
        
        int filled_count = p.count_filled();
        score -= (81 - filled_count) * 2.0; // 空位惩罚

        if (p.has_7x7_bonus) score += 7.0;

        // 纽扣收入预估：根据 income_triggers 剩余数量
        int triggers_left = 0;
        for (int t : PatchworkGame::income_triggers) {
            if (p.time_pos < t) triggers_left++;
        }
        score += p.income * triggers_left;

        return score;
    }

    // 获取带权重的合法动作列表
    std::vector<std::pair<Action, double>> get_prior_actions(const PatchworkGame& game) {
        auto actions = game.get_legal_actions();
        std::vector<std::pair<Action, double>> res;
        res.reserve(actions.size());
        
        double sum = 0.0;
        for(const auto& a : actions) {
            double w = get_action_weight(game, a);
            res.push_back({a, w});
            sum += w;
        }
        for(auto& pair : res) pair.second /= (sum + 1e-9);
        return res;
    }

    // MCTS 工作线程
    SmartMCTSNode* run_mcts_worker(PatchworkGame real_game, double limit, int seed) {
        std::mt19937 local_rng(seed);
        // 根节点的 action.type = -1 代表初始状态
        SmartMCTSNode* root = new SmartMCTSNode(nullptr, {-1, 0, 0, -1}, real_game.current_player_idx, 1.0);
        root->untried_actions = get_prior_actions(real_game);
        
        auto start_time = std::chrono::steady_clock::now();
        int sim_count = 0;

        while (true) {
            if ((sim_count & 127) == 0) { // 减少检查时间的频率
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (std::chrono::duration<double>(elapsed).count() >= limit) break;
            }

            PatchworkGame sim_game = real_game;
            SmartMCTSNode* node = root;

            // 1. Selection (PUCT)
            while (node->untried_actions.empty() && !node->children.empty() && !sim_game.game_over) {
                bool is_me = (sim_game.current_player_idx == root->player_idx);
                double best_s = is_me ? -1e18 : 1e18;
                SmartMCTSNode* best_c = nullptr;

                double sqrt_parent_visits = std::sqrt(node->visits);
                for (auto* child : node->children) {
                    double q = (child->visits > 0) ? (child->value / child->visits) : 0.0;
                    // PUCT 公式
                    double u = 1.41 * child->prior_prob * sqrt_parent_visits / (1 + child->visits);
                    double score = is_me ? (q + u) : (q - u);
                    if (is_me ? (score > best_s) : (score < best_s)) { 
                        best_s = score; 
                        best_c = child; 
                    }
                }
                if (!best_c) break;
                node = best_c;
                sim_game.apply_action(node->action);
            }

            // 2. Expansion
            if (!node->untried_actions.empty() && !sim_game.game_over) {
                // 弹出最后一个动作（效率比 erase 随机位置高）
                auto act_pair = node->untried_actions.back();
                node->untried_actions.pop_back();
                
                sim_game.apply_action(act_pair.first);
                SmartMCTSNode* child = new SmartMCTSNode(node, act_pair.first, sim_game.current_player_idx, act_pair.second);
                
                if (!sim_game.game_over) child->untried_actions = get_prior_actions(sim_game);
                node->children.push_back(child);
                node = child;
            }

            // 3. Rollout (快速模拟)
            int depth = 0;
            while (!sim_game.game_over && depth < max_depth) {
                auto acts = sim_game.get_legal_actions();
                if (acts.empty()) break;
                // 模拟阶段使用随机策略，或简单的贪心
                std::uniform_int_distribution<> d(0, (int)acts.size() - 1);
                sim_game.apply_action(acts[d(local_rng)]);
                depth++;
            }

            // 4. Backpropagation
            double my_score = evaluate_potential(sim_game, root->player_idx);
            double op_score = evaluate_potential(sim_game, 1 - root->player_idx);
            // 将分数差距映射到 [-1, 1]
            double reward = std::tanh((my_score - op_score) / 40.0);

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

    Action get_best_action(PatchworkGame real_game) {
        auto start_t = std::chrono::steady_clock::now();
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads == 0) n_threads = 4;
        
        // 分配任务
        std::vector<std::future<SmartMCTSNode*>> futures;
        double thread_limit = time_limit * 0.95; // 留出一点汇总时间
        for (unsigned int i = 0; i < n_threads; ++i) {
            futures.push_back(std::async(std::launch::async, &SmartPatchworkAI::run_mcts_worker, this, real_game, thread_limit, (int)std::random_device{}()));
        }

        // 汇总结果
        struct Stats { int v = 0; double val = 0; Action act; };
        std::vector<Stats> final_stats;

        for (auto& f : futures) {
            SmartMCTSNode* t_root = f.get();
            for (auto* child : t_root->children) {
                bool found = false;
                for (auto& s : final_stats) {
                    // 动作相同判定：类型相同、如果是买块则ID相同、Mask相同
                    if (s.act.type == child->action.type && 
                        s.act.piece_id == child->action.piece_id && 
                        s.act.mask == child->action.mask) {
                        s.v += child->visits;
                        s.val += child->value;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    final_stats.push_back({child->visits, child->value, child->action});
                }
            }
            delete t_root;
        }

        // 选择访问次数最多的动作
        std::sort(final_stats.begin(), final_stats.end(), [](const Stats& a, const Stats& b){
            return a.v > b.v;
        });

        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_t).count();
        if (!final_stats.empty()) {
            std::cout << "\033[36m[AI 决策] 耗时:" << std::fixed << std::setprecision(2) << elapsed 
                      << "s | 总模拟次数:" << (final_stats[0].v * n_threads) // 估算
                      << " | 最佳动作访问量:" << final_stats[0].v << "\033[0m" << std::endl;
            return final_stats[0].act;
        }

        // 保底：万一没搜出来（极少见），随机给个动作
        return real_game.get_legal_actions()[0];
    }
};