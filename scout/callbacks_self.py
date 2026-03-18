import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ScoutWinRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=100, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env # 注意：eval_env.unwrapped 也要有 opponent_model
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_score_diff = -99.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            wins = 0
            total_player_scores = np.zeros(4)

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                while not done:
                    # P0 由当前正在训练的 self.model 控制
                    mask = self.eval_env.get_wrapper_attr("action_masks")()
                    action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                
                game = self.eval_env.unwrapped.game
                scores = [game.round_scores.get(i, 0) for i in range(4)]
                for i in range(4): total_player_scores[i] += scores[i]
                if scores[0] > max(scores[1:]): wins += 1

            avg_win_rate = wins / self.n_eval_episodes
            avg_player_scores = total_player_scores / self.n_eval_episodes
            avg_score_diff = avg_player_scores[0] - np.mean(avg_player_scores[1:])

            if self.verbose > 0:
                print(f"\n[Self-Play 评估] Step: {self.num_timesteps}")
                print(f"胜率 (vs 当前对手): {avg_win_rate:.2%} | 平均分差: {avg_score_diff:+.2f}")
            
            # 只有当显著优于当前的对手时，才保存“最佳进化版”
            if avg_score_diff > self.best_score_diff and avg_score_diff > 0.5:
                self.best_score_diff = avg_score_diff
                save_path = f"./models/best_evolution_{self.num_timesteps}"
                self.model.save(save_path)
                print(f"🌟 发现更强的进化版本，已保存！")

        return True

class SelfPlayUpdateCallback(BaseCallback):
    def __init__(self, update_freq: int, save_path: str, eval_env=None, verbose=1):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.eval_env = eval_env # 同时更新评估环境里的对手

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            # 1. 保存当前版本
            path = os.path.join(self.save_path, f"opponent_step_{self.num_timesteps}.zip")
            self.model.save(path)
            
            # 2. 更新训练环境 (VecEnv 所有的子环境都要更)
            self.training_env.env_method("set_opponent_model", self.model)
            
            # 3. 更新评估环境
            if self.eval_env is not None:
                # 如果 eval_env 被包裹了多层，记得 unwrapped
                self.eval_env.unwrapped.set_opponent_model(self.model)
            
            if self.verbose > 0:
                print(f"\n🔄 [Self-Play] 对手等级提升！已将当前模型设为 P1-P3 的策略。")
        return True