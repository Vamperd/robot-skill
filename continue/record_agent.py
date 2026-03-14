import argparse

import imageio
import numpy as np
import pygame
from stable_baselines3 import PPO
from mutiple_train import LTLfGymEnv
from transfer_eval import EVAL_SCENARIOS, resolve_observation_mode

def get_scenario_by_name(scenario_name):
    for scenario in EVAL_SCENARIOS:
        if scenario["name"] == scenario_name:
            return scenario
    available = ", ".join(item["name"] for item in EVAL_SCENARIOS)
    raise ValueError(f"未找到场景 '{scenario_name}'，可选场景: {available}")


def record_gif(model_path, scenario_name="baseline_train_layout", gif_name="agent_trajectory.gif", fps=60, observation_mode=None):
    print(f"正在加载模型: {model_path} ...")
    model = PPO.load(model_path, device="cpu")
    observation_mode = resolve_observation_mode(model, observation_mode)
    scenario = get_scenario_by_name(scenario_name)

    # 初始化环境，必须开启 human 渲染模式才能抓取画面
    env = LTLfGymEnv(
        render_mode="human",
        obstacles=scenario["obstacles"],
        tasks=scenario["tasks"],
        start_pos=scenario.get("start_pos", (100.0, 100.0)),
        max_steps=scenario.get("max_steps", 2000),
        observation_mode=observation_mode,
    )
    
    obs, _ = env.reset()
    frames = []
    
    print(f"开始录制场景: {scenario_name}，请不要关闭弹出的 Pygame 窗口...")
    # 给它最多 2000 步的时间跑完
    for step in range(env.max_steps):
        # 让 Pygame 处理事件，防止窗口卡死
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 模型预测动作 (deterministic=True 保证每次回放表现最稳定)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 必须调用 render 刷新画面
        env.render()
        
        # --- 核心：截取 Pygame 画面并转换为图像帧 ---
        # 1. 获取当前屏幕的 3D 像素数组
        frame_3d = pygame.surfarray.array3d(env.screen)
        # 2. Pygame 的坐标系是 (Width, Height, RGB)，而图片格式需要 (Height, Width, RGB)
        # 所以我们需要用 numpy 做一个轴的转置 (Transpose)
        frame_3d = np.transpose(frame_3d, (1, 0, 2))
        frames.append(frame_3d)
        
        # 如果任务完成或者超时，停止录制
        if terminated or truncated:
            print(f"回合结束！(共耗时 {step} 步)")
            break

    env.close()
    
    print(f"正在将 {len(frames)} 帧画面打包导出为 {gif_name} ...")
    # 生成 GIF 动图，duration 是每帧的持续时间 (毫秒)
    imageio.mimsave(gif_name, frames, duration=1000/fps)
    print("导出成功！你可以去文件夹里用浏览器或看图软件反复观看了。")


def main():
    parser = argparse.ArgumentParser(description="录制强化学习策略在指定场景中的回放 GIF")
    parser.add_argument("--model-path", default="generalization_eval_best_v3/best_model.zip", help="模型路径")
    parser.add_argument("--scenario", default="hard_ood_layout", help="要回放的场景名")
    parser.add_argument("--gif-name", default="agent_trajectory.gif", help="输出 GIF 文件名")
    parser.add_argument("--fps", type=int, default=60, help="GIF 帧率")
    parser.add_argument("--observation-mode", choices=["absolute", "relative"], default=None, help="观测模式，不填时自动推断")
    args = parser.parse_args()

    record_gif(
        model_path=args.model_path,
        scenario_name=args.scenario,
        gif_name=args.gif_name,
        fps=args.fps,
        observation_mode=args.observation_mode,
    )


if __name__ == "__main__":
    main()