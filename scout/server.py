from flask import Flask, jsonify, request, render_template
import torch as th
import numpy as np
import os
from sb3_contrib import MaskablePPO
from gym_env import ScoutEnv  # 确保文件名匹配

app = Flask(__name__, static_folder="static", template_folder="templates")

# ==========================================
# 1. 环境与模型初始化
# ==========================================
env = ScoutEnv()
# 建议检查路径是否存在
MODEL_PATH = "/data/feiyu/code/boardGameAI/scout/models/best_diff_model_4360000.zip"

if os.path.exists(MODEL_PATH):
    model = MaskablePPO.load(MODEL_PATH)
else:
    print(f"警告: 找不到模型文件 {MODEL_PATH}")
    model = None

obs, info = env.reset()

# ==========================================
# 2. 路由定义
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    game = env.unwrapped.game
    is_setup_phase = (game.setup_choices[0] == -1)
    
    # 核心修改：确保 table_cards 里的每一张牌都是 (正面, 反面) 的元组
    # 注意：如果你的游戏引擎在 table_cards 里只存了数字，
    # 你可能需要从牌堆或初始数据中获取它们的原始元组。
    # 这里假设 game.table_cards 已经存储了元组数据。
    
    state = {
        "phase": game.phase,
        "is_setup_phase": is_setup_phase,
        "current_player": game.current_player,
        "table_cards": game.table_cards,  # 确保这里是 [(val1, val2), ...]
        "table_owner": game.table_owner,
        "human_hand_raw": game.hands[0],
        "setup_choice": game.setup_choices[0],
        "npc_card_counts": {i: len(game.hands[i]) for i in range(1, 4)},
    }
    return jsonify(state)

# 新增：初始选择手牌方向
@app.route('/api/choose_side', methods=['POST'])
def choose_side():
    data = request.json
    side = int(data.get('side')) # 0 或 1
    game = env.unwrapped.game
    game.setup_choices[0] = side
    # 如果你的环境逻辑里有特定的 Phase 切换，可以在这里处理
    return jsonify({"status": "success"})

@app.route('/api/human_action', methods=['POST'])
def human_action():
    """处理人类玩家动作 (仅此一份，已修复重复命名冲突)"""
    global obs
    data = request.json
    action_type = data.get('type')
    game = env.unwrapped.game
    
    action_idx = None
    legal_actions = game.get_legal_actions(0)
    
    try:
        if action_type == 'SHOW':
            start = int(data.get('start'))
            length = int(data.get('length'))
            action_idx = game.OFFSET_SHOW + game._encode_show(start, length)
            
        elif action_type in ['SCOUT', 'SCOUT_SHOW']:
            side = int(data.get('side'))
            insert_idx = int(data.get('insert_idx'))
            flip = int(data.get('flip'))
            code = game._encode_scout(side, insert_idx, flip)
            
            if action_type == 'SCOUT':
                action_idx = game.OFFSET_SCOUT + code
            else:
                action_idx = game.OFFSET_SCOUT_SHOW + code
                
        elif action_type == 'PASS':
            action_idx = game.ACTION_PASS

        # 校验合法性
        if action_idx not in legal_actions:
            return jsonify({"status": "error", "message": "非法动作！"}), 400
            
        obs, reward, done, truncated, info = env.step(action_idx)
        return jsonify({"status": "success", "done": done})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/ai_step', methods=['POST'])
def ai_step():
    """驱动 AI 或 NPC 运行"""
    global obs
    game = env.unwrapped.game
    
    if game.done:
        return jsonify({"status": "game_over"})
        
    # 如果当前轮到 NPC (P1, P2, P3)
    if game.current_player != 0:
        action_masks = env.unwrapped._gen_mask()
        with th.no_grad():
            obs_tensor = th.as_tensor(obs).unsqueeze(0).to(model.device) 
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]
            masked_probs = (probs * action_masks) / ((probs * action_masks).sum() + 1e-8)
            action_idx = np.argmax(masked_probs)
        
        obs, reward, done, truncated, info = env.step(int(action_idx))
        
    return jsonify({"status": "success"})

def process_scout(side, insert_idx, flip):
    """
    side: 0 或 1 (拿桌面左边还是右边)
    insert_idx: 插入手牌的物理位置 (0 到 len(hand))
    flip: 0 或 1 (是否翻转)
    """
    game = env.unwrapped.game
    
    # 1. 编码动作
    # 这里的编码逻辑必须和你的环境完全一致
    code = game._encode_scout(side, insert_idx, flip)
    action_idx = game.OFFSET_SCOUT + code
    
    # 2. 执行环境步进
    obs, reward, done, _, info = env.step(action_idx)
    return action_idx

if __name__ == '__main__':
    # 使用 0.0.0.0 可以让你在局域网内用手机或其他电脑测试
    app.run(host='0.0.0.0', debug=True, port=5000)