import argparse

import imageio
import numpy as np
import pygame
from stable_baselines3 import PPO

from transfer_eval import EVAL_SCENARIOS, get_base_env, make_vec_env, resolve_observation_config


def get_scenario_by_name(scenario_name):
    for scenario in EVAL_SCENARIOS:
        if scenario["name"] == scenario_name:
            return scenario
    available = ", ".join(item["name"] for item in EVAL_SCENARIOS)
    raise ValueError(f"未找到场景 '{scenario_name}'，可选场景: {available}")


def record_gif(model_path, scenario_name="baseline_train_layout", gif_name="agent_trajectory.gif", fps=60, observation_mode=None, n_stack=None):
    print(f"正在加载模型: {model_path} ...")
    model = PPO.load(model_path, device="cpu")
    observation_mode, n_stack = resolve_observation_config(model, observation_mode, n_stack)
    scenario = get_scenario_by_name(scenario_name)

    env = make_vec_env(scenario, observation_mode=observation_mode, n_stack=n_stack, render=True)
    base_env = get_base_env(env)
    obs = env.reset()
    frames = []

    print(f"开始录制场景: {scenario_name}，观测模式={observation_mode}，n_stack={n_stack}，请不要关闭弹出的 Pygame 窗口...")
    for step in range(base_env.max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        env.render()

        frame_3d = pygame.surfarray.array3d(base_env.screen)
        frame_3d = np.transpose(frame_3d, (1, 0, 2))
        frames.append(frame_3d)

        if done[0]:
            print(f"回合结束！(共耗时 {step} 步)")
            break

    env.close()

    print(f"正在将 {len(frames)} 帧画面打包导出为 {gif_name} ...")
    imageio.mimsave(gif_name, frames, duration=1000 / fps)
    print("导出成功！你可以去文件夹里用浏览器或看图软件反复观看了。")


def main():
    parser = argparse.ArgumentParser(description="录制强化学习策略在指定场景中的回放 GIF")
    parser.add_argument("--model-path", default="generalization_eval_best_vel_punishment/best_model.zip", help="模型路径")
    parser.add_argument("--scenario", default="ultimate_generalization_test", help="要回放的场景名")
    parser.add_argument("--gif-name", default="agent_trajectory.gif", help="输出 GIF 文件名")
    parser.add_argument("--fps", type=int, default=60, help="GIF 帧率")
    parser.add_argument("--observation-mode", choices=["absolute", "relative"], default=None, help="观测模式，不填时自动推断")
    parser.add_argument("--n-stack", type=int, default=None, help="Frame stack 数量；不填时自动根据模型推断")
    args = parser.parse_args()

    record_gif(
        model_path=args.model_path,
        scenario_name=args.scenario,
        gif_name=args.gif_name,
        fps=args.fps,
        observation_mode=args.observation_mode,
        n_stack=args.n_stack,
    )


if __name__ == "__main__":
    main()
