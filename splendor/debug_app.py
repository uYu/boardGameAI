import streamlit as st
import numpy as np
import pandas as pd
from gym_env import SplendorEnv

# ================= 配置与常量 =================
st.set_page_config(layout="wide", page_title="Splendor Env Matrix Debugger")
ALL_RESOURCES = ['white', 'blue', 'green', 'red', 'black', 'gold']

# ================= 矩阵还原函数 =================
def get_raw_matrix_df(flat_mat):
    """将 42 维向量 reshape 回 7x6 的 0-1 矩阵"""
    mat = flat_mat.reshape(7, 6)
    df = pd.DataFrame(
        mat, 
        columns=['White', 'Blue', 'Green', 'Red', 'Black', 'Gold'],
        index=[f'数量 {i+1}' for i in range(7)]
    )
    return df

def decode_card_feat_raw(card_feat):
    """解析 50 维卡牌特征：保留原汁原味的 0-1 矩阵和标量"""
    if np.sum(card_feat) == 0:
        return None
    
    # 前 42 维是成本矩阵
    cost_df = get_raw_matrix_df(card_feat[:42])
    
    # 后 8 维是标量
    pts = int(card_feat[42])
    color_onehot = card_feat[43:48].tolist()
    level = int(card_feat[48])
    can_buy = int(card_feat[49])
    
    scalars = {
        "分数 (1)": pts,
        "颜色 One-hot (5)": str(color_onehot),
        "等级 (1)": level,
        "买得起 (1)": can_buy
    }
    return cost_df, scalars

# ================= 初始化环境 =================
if 'env' not in st.session_state:
    st.session_state.env = SplendorEnv()
    st.session_state.obs, _ = st.session_state.env.reset()

env = st.session_state.env

# ================= 侧边栏 =================
with st.sidebar:
    st.header("🎮 控制台")
    if st.button("🔄 重置到初始局面", use_container_width=True):
        st.session_state.obs, _ = env.reset()
        
    if st.button("🎲 随机走走 (生成随机场景)", use_container_width=True):
        st.session_state.obs, _ = env.reset()
        for _ in range(np.random.randint(10, 30)):
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                obs, reward, term, trunc, info = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
                    break
        st.session_state.obs = obs

obs = st.session_state.obs

# ================= 主页面：状态展示 =================
st.title("💎 Splendor 特征矩阵查错器")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🧠 全局观测矩阵 (Matrix View)")
    
    st.subheader("🏦 银行宝石矩阵 (Bank - 42 维 -> 7x6)")
    st.markdown("每一列代表一种颜色，从上到下按数量填充 `1`。")
    # 使用 st.dataframe 的高亮功能，让 1 更明显
    bank_df = get_raw_matrix_df(obs["bank"])
    st.dataframe(bank_df.style.highlight_max(axis=None, color='#5a9e5a'), use_container_width=True)
    
    st.subheader("👤 玩家状态 (Player Stats)")
    st.markdown("**你的宝石矩阵 (42 维 -> 7x6)**")
    p_gem_df = get_raw_matrix_df(obs["player_stats"][:42])
    st.dataframe(p_gem_df.style.highlight_max(axis=None, color='#5a9e5a'), use_container_width=True)
    
    st.markdown("**你的标量特征 (6 维)**")
    p_scalars = {ALL_RESOURCES[i] + " Bonus": obs["player_stats"][42+i] for i in range(5)}
    p_scalars["Score"] = obs["player_stats"][47]
    st.json(p_scalars)
with col2:
    st.header("🃏 详细矩阵分布 (Full Matrix Inspect)")

    # 使用 Tabs 节省空间，同时展示所有牌
    tab1, tab2, tab3, tab4 = st.tabs(["场上卡牌 (12)", "预留卡牌 (3)", "贵族 (3)", "对手状态"])

    with tab1:
        st.markdown("检查 12 张场上牌的 50 维特征")
        for i in range(12):
            with st.expander(f"Level {i//4 + 1} - 槽位 {i%4} 的矩阵"):
                card_feat = obs["board_cards"][i]
                parsed = decode_card_feat_raw(card_feat)
                if parsed:
                    cost_df, scalars = parsed
                    st.dataframe(cost_df.style.highlight_max(axis=None, color='#5a9e5a'))
                    st.json(scalars)
                else:
                    st.text("空槽位 (全 0)")

    with tab2:
        st.markdown("检查你自己预留的 3 张牌")
        for i in range(3):
            with st.expander(f"预留槽位 {i}"):
                res_feat = obs["reserved_cards"][i]
                parsed = decode_card_feat_raw(res_feat)
                if parsed:
                    cost_df, scalars = parsed
                    st.dataframe(cost_df.style.highlight_max(axis=None, color='#3498db'))
                    st.json(scalars)
                else:
                    st.text("未预留卡牌")

    with tab3:
        st.markdown("检查 3 个贵族需求 (42 维矩阵)")
        for i in range(3):
            with st.expander(f"贵族 {i+1}"):
                noble_mat = obs["nobles"][i]
                if np.sum(noble_mat) > 0:
                    noble_df = get_raw_matrix_df(noble_mat)
                    st.dataframe(noble_df.style.highlight_max(axis=None, color='#f1c40f'))
                else:
                    st.text("无贵族")

    with tab4:
        st.markdown("**对手的宝石矩阵 (42 维)**")
        opp_gem_df = get_raw_matrix_df(obs["opponent_stats"][:42])
        st.dataframe(opp_gem_df.style.highlight_max(axis=None, color='#e74c3c'))
        
        st.markdown("**对手的标量特征 (7 维)**")
        # 42(矩阵) + 5(Bonus) + 1(Score) + 1(ReservedCount) = 49
        o_scalars = {ALL_RESOURCES[i] + " Bonus": obs["opponent_stats"][42+i] for i in range(5)}
        o_scalars["Score"] = obs["opponent_stats"][47]
        o_scalars["Reserved Count"] = obs["opponent_stats"][48]
        st.json(o_scalars)
        
st.divider()

# ================= 动作检查 =================
st.header("⚡ 合法动作 (Action Masks)")
masks = env.action_masks()
valid_indices = np.where(masks)[0]

if len(valid_indices) == 0:
    st.warning("当前没有任何合法动作！")
else:
    action_data = []
    for idx in valid_indices:
        decoded = env._decode_action(idx)
        action_data.append({"Action ID (0-44)": idx, "解码动作": str(decoded)})
    st.dataframe(pd.DataFrame(action_data), hide_index=True)