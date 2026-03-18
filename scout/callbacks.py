import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ScoutWinRateCallback(BaseCallback):
    """
    评估 Agent 的胜率、平均强化学习奖励以及平均局内原始得分
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=100, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_win_rate = -1.0
        self.best_score_diff = -10.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            wins = 0
            total_rl_reward = 0
            # 使用数组来记录 4 个玩家的总分，方便计算平均值
            total_player_scores = np.zeros(4, dtype=np.float32) 

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                episode_rl_reward = 0

                while not done:
                    action_masks = self.eval_env.get_wrapper_attr("action_masks")()
                    action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_rl_reward += reward
                    done = terminated or truncated

                # --- 核心统计逻辑 ---
                game = self.eval_env.unwrapped.game
                # 累加每个玩家的得分
                for i in range(4):
                    total_player_scores[i] += game.round_scores.get(i, 0)

                # 判断胜率 (AI 是 P0)
                p0_score = game.round_scores.get(0, 0)
                others_scores = [game.round_scores.get(i, 0) for i in range(1, 4)]
                if p0_score > max(others_scores):
                    wins += 1

                total_rl_reward += episode_rl_reward

            # 计算平均值
            avg_player_scores = total_player_scores / self.n_eval_episodes
            avg_win_rate = wins / self.n_eval_episodes
            avg_rl_reward = total_rl_reward / self.n_eval_episodes

            # 计算 AI 相对于其他人的平均分差
            avg_score_diff = avg_player_scores[0] - np.mean(avg_player_scores[1:])

            # 打印到控制台
            if self.verbose > 0:
                print(f"\n" + "="*50)
                print(f"[评估] Step: {self.num_timesteps}")
                print(f"平均分差 (AI - Opponents): {avg_score_diff:+.2f} ★")
                print(f"胜率: {avg_win_rate:.2%}")

                # 重点修改：打印所有玩家的平均得分
                avg_scores_str = ", ".join([f"P{i}: {avg_player_scores[i]:.2f}" for i in range(4)])
                print(f"各玩家平均得分: [{avg_scores_str}]")

                print(f"评估场次平均 RL Reward: {avg_rl_reward:.2f}")
                print("="*50)

            # 记录到 TensorBoard
            self.logger.record("eval/avg_score_diff", avg_score_diff)
            self.logger.record("eval/win_rate", avg_win_rate)
            for i in range(4):
                self.logger.record(f"eval/avg_score_p{i}", avg_player_scores[i])

            # 保存逻辑
            if avg_score_diff > self.best_score_diff:
                self.best_score_diff = avg_score_diff
                if avg_score_diff > 0:
                    save_path = f"./models/best_diff_model_{self.num_timesteps}"
                    self.model.save(save_path)
                    print(f"检测到历史最高分差！模型已保存。")

        return True

class SelfPlayUpdateCallback(BaseCallback):
    def __init__(self, update_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def _on_step(self) -> bool:
        # 每隔 update_freq 步更新一次对手
        if self.n_calls % self.update_freq == 0:
            path = os.path.join(self.save_path, "latest_opponent.zip")
            self.model.save(path)
            
            # 更新训练环境中的对手 (env 是 ActionMasker，所以要访问 .unwrapped)
            if hasattr(self.training_env.envs[0].unwrapped, 'opponent_policy'):
                # 重新加载策略给对手使用
                self.training_env.envs[0].unwrapped.opponent_policy = self.model
            
            if self.verbose > 0:
                print(f"--- [Self-Play] 对手模型已更新 (Step: {self.num_timesteps}) ---")
        return True