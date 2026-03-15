import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from local_rl_env import LocalRLEnv

def make_env(rank: int, base_seed: int = 42):
    def _init():
        env = LocalRLEnv(render_mode="None", max_steps=500)
        env.reset(seed=base_seed + rank)
        return Monitor(env)
    return _init

class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.03, final_ent_coef=0.0001, total_timesteps=1_500_000, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        current_ent_coef = self.initial_ent_coef - progress * (self.initial_ent_coef - self.final_ent_coef)
        self.model.ent_coef = current_ent_coef
        self.logger.record("config/ent_coef", current_ent_coef)
        return True

def main():
    num_envs = 8
    total_timesteps = 1_500_000

    vec_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.03,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        tensorboard_log="./ppo_local_planner_tensorboard/",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./local_eval_best/",
        log_path="./local_eval_logs/",
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )
    
    ent_decay_callback = EntropyDecayCallback(
        initial_ent_coef=0.03, 
        final_ent_coef=0.0001, 
        total_timesteps=total_timesteps
    )
    
    callback_list = CallbackList([eval_callback, ent_decay_callback])

    try:
        print("开始训练底层局部导航 RL Agent... (按 Ctrl+C 可中断)")
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback_list)
    except KeyboardInterrupt:
        print("\n[警告] 训练被手动中断！正在保存当前最新模型...")
    except Exception as e:
        print(f"\n[错误] 训练发生异常: {e}")
    finally:
        save_path = "ppo_local_planner"
        model.save(save_path)
        print(f"局部避障规划器模型已保存至: {save_path}.zip")
        
        vec_env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
