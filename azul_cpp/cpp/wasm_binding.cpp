#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "AzulGame.hpp"
#include "SmartAzulAI.hpp"

using namespace emscripten;

// --- 包装函数：将复杂的 C++ 接口转化为简单的全局函数供 JS 调用 ---

// 创建游戏实例并返回指针
AzulGame* createGame(int num_players, unsigned int seed) {
    return new AzulGame(num_players, seed);
}

// 销毁游戏实例（防止 Wasm 内存泄漏）
void deleteGame(AzulGame* game) {
    if (game) delete game;
}

// 执行动作：接收原始参数，内部构造 Action 并执行，最后返回最新的状态 JSON
std::string applyMove(AzulGame& game, int src, int col, int row) {
    Action a;
    a.source_id = (int8_t)src;
    a.color = (int8_t)col;
    a.target_row = (int8_t)row;
    
    game.step(a);
    return game.to_json();
}

// 调用 AI：传入时间限制，返回 AI 的思考结果 JSON
std::string askAI(AzulGame& game, double time_limit) {
    SmartAzulAI ai(time_limit);
    AIDecision decision = ai.get_best_decision(game);
    
    std::stringstream ss;
    ss << "{";
    ss << "\"best_action\":{\"src\":" << (int)decision.best_action.source_id 
       << ",\"col\":" << (int)decision.best_action.color 
       << ",\"row\":" << (int)decision.best_action.target_row << "},";
    ss << "\"win_rate\":" << decision.win_rate << ",";
    ss << "\"sims\":" << decision.total_sims << ",";
    ss << "\"time\":" << decision.elapsed_time;
    ss << "}";
    return ss.str();
}

// --- Emscripten 绑定模块 ---

EMSCRIPTEN_BINDINGS(azul_module) {
    // 1. 绑定 AzulGame 类，使其能在 JS 中作为对象存在
    class_<AzulGame>("AzulGame")
        .function("toJSON", &AzulGame::to_json);

    // 2. 绑定全局工具函数
    // allow_raw_pointers 允许 JS 获取 createGame 返回的指针
    function("createGame", &createGame, allow_raw_pointers());
    function("deleteGame", &deleteGame, allow_raw_pointers());
    function("applyMove", &applyMove);
    function("askAI", &askAI);
}

#include <emscripten.h>

int main() {
    // 告诉 Emscripten，主循环结束后不要关闭环境
    EM_ASM({
        console.log("Wasm 线程池已在后台准备就绪");
    });
    return 0;
}