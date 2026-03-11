import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.utils import get_action_masks
# 必须导入你的策略，否则回调函数无法让对手走棋
from policy import HeuristicPolicy 

class SplendorEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=100, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_win_rate = -1.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            wins = 0
            total_scores = np.zeros(2)

            for _ in range(self.n_eval_episodes):
                # 每局开始前必须 reset
                obs, _ = self.eval_env.reset()
                done = False
                step_count = 0
                max_eval_steps = 50  # 限制最多打 50 手

                while not done and step_count < max_eval_steps:
                    action_masks = get_action_masks(self.eval_env)
                    action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                    
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    step_count += 1

                # --- 核心：强制结算逻辑 ---
                game = self.eval_env.unwrapped.game
                p0_score = game.players[0].score
                p1_score = game.players[1].score
                
                total_scores[0] += p0_score
                total_scores[1] += p1_score

                # 判定胜负：谁分高谁赢
                if p0_score > p1_score:
                    wins += 1
                # 如果平分，通常规则是卡牌少的赢，这里简单处理
                elif p0_score == p1_score:
                    if len(game.players[0].reserved) < len(game.players[1].reserved):
                        wins += 1

            # --- 修复 RuntimeError 的关键：评估结束后重置环境 ---
            self.eval_env.reset()

            avg_win_rate = wins / self.n_eval_episodes
            avg_scores = total_scores / self.n_eval_episodes
            
            if self.verbose > 0:
                print(f"\n[Step {self.num_timesteps}] 评估完成 (限 50 步):")
                print(f"胜率: {avg_win_rate:.2%} | AI均分: {avg_scores[0]:.2f} | 脚本均分: {avg_scores[1]:.2f}")

            # 记录到 Tensorboard
            self.logger.record("eval/win_rate", avg_win_rate)
            self.logger.record("eval/score_diff", avg_scores[0] - avg_scores[1])
            
        return True