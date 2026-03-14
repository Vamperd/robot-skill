import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mutiple_train import LTLfGymEnv


EVAL_SCENARIOS = [
    {
        "name": "baseline_train_layout",
        "start_pos": (100.0, 100.0),
        "obstacles": [
            (200, 150, 50, 300),
            (450, 250, 200, 50),
            (500, 400, 50, 200),
        ],
        "tasks": {
            "Task A": (100, 500),
            "Task B": (400, 100),
            "Task C": (700, 500),
        },
    },
    {
        "name": "shift_tasks_only",
        "start_pos": (100.0, 100.0),
        "obstacles": [
            (200, 150, 50, 300),
            (450, 250, 200, 50),
            (500, 400, 50, 200),
        ],
        "tasks": {
            "Task A": (160, 500),
            "Task B": (360, 160),
            "Task C": (650, 460),
        },
    },
    {
        "name": "shift_obstacles_only",
        "start_pos": (100.0, 100.0),
        "obstacles": [
            (240, 120, 50, 280),
            (420, 300, 220, 50),
            (540, 380, 50, 180),
        ],
        "tasks": {
            "Task A": (100, 500),
            "Task B": (400, 100),
            "Task C": (700, 500),
        },
    },
    {
        "name": "shift_both_layout",
        "start_pos": (120.0, 80.0),
        "obstacles": [
            (180, 180, 60, 260),
            (420, 220, 180, 60),
            (560, 360, 60, 200),
        ],
        "tasks": {
            "Task A": (160, 480),
            "Task B": (500, 120),
            "Task C": (680, 420),
        },
    },
    {
        "name": "hard_ood_layout",
        "start_pos": (80.0, 80.0),
        "obstacles": [
            (150, 120, 80, 320),
            (320, 260, 260, 60),
            (620, 100, 50, 280),
        ],
        "tasks": {
            "Task A": (120, 520),
            "Task B": (520, 80),
            "Task C": (720, 520),
        },
    },
]


def run_episode(model, scenario, render=False, observation_mode="absolute"):
    env = LTLfGymEnv(
        render_mode="human" if render else "None",
        obstacles=scenario["obstacles"],
        tasks=scenario["tasks"],
        start_pos=scenario.get("start_pos", (100.0, 100.0)),
        max_steps=scenario.get("max_steps", 2000),
        observation_mode=observation_mode,
    )
    obs, _ = env.reset()
    total_reward = 0.0
    success = False
    steps = 0

    for steps in range(1, env.max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if terminated:
            success = True
            break
        if truncated:
            break

    env.close()
    return {
        "success": success,
        "steps": steps,
        "total_reward": float(total_reward),
        "final_dfa_state": int(env.dfa.state),
    }


def resolve_observation_mode(model, observation_mode=None):
    obs_shape = getattr(model.observation_space, "shape", None)
    obs_dim = obs_shape[0] if obs_shape else None
    mode_to_dims = {"absolute": {13, 21}, "relative": {17, 25}}

    if observation_mode is None:
        if obs_dim in mode_to_dims["absolute"]:
            return "absolute"
        if obs_dim in mode_to_dims["relative"]:
            return "relative"
        raise ValueError(f"无法根据模型观测维度 {obs_dim} 自动判断 observation_mode")

    expected_dims = mode_to_dims[observation_mode]
    if obs_dim not in expected_dims:
        raise ValueError(
            f"模型期望观测维度为 {obs_dim}，但当前 observation_mode='{observation_mode}' 允许的维度为 {sorted(expected_dims)}。"
            f"请改用 observation_mode='{'relative' if obs_dim in mode_to_dims['relative'] else 'absolute'}'。"
        )
    return observation_mode


def evaluate(model_path, render_first=False, observation_mode="absolute"):
    model = PPO.load(model_path, device="cpu")
    observation_mode = resolve_observation_mode(model, observation_mode)
    results = []

    for idx, scenario in enumerate(EVAL_SCENARIOS):
        result = run_episode(model, scenario, render=render_first and idx == 0, observation_mode=observation_mode)
        result["scenario"] = scenario["name"]
        results.append(result)

    success_rate = np.mean([item["success"] for item in results])
    avg_reward = np.mean([item["total_reward"] for item in results])
    avg_final_dfa = np.mean([item["final_dfa_state"] for item in results])

    summary = {
        "model_path": str(model_path),
        "success_rate": float(success_rate),
        "avg_reward": float(avg_reward),
        "avg_final_dfa_state": float(avg_final_dfa),
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="评估 PPO 模型在新障碍/目标布局下的迁移能力")
    parser.add_argument("--model-path", default="generalization_eval_best_v3/best_model.zip", help="已训练模型路径")
    parser.add_argument("--observation-mode", choices=["absolute", "relative"], default=None, help="评估时使用的观测模式；不填时自动根据模型推断")
    parser.add_argument("--render-first", action="store_true", help="渲染第一个测试场景")
    parser.add_argument("--save-json", default="transfer_eval_results.json", help="结果保存路径")
    args = parser.parse_args()

    summary = evaluate(args.model_path, render_first=args.render_first, observation_mode=args.observation_mode)

    print("=== 迁移测试结果 ===")
    print(f"模型: {summary['model_path']}")
    print(f"成功率: {summary['success_rate']:.2%}")
    print(f"平均回报: {summary['avg_reward']:.2f}")
    print(f"平均最终 DFA 状态: {summary['avg_final_dfa_state']:.2f} / 3")
    print()

    for item in summary["results"]:
        print(
            f"- {item['scenario']}: "
            f"success={item['success']}, "
            f"steps={item['steps']}, "
            f"reward={item['total_reward']:.2f}, "
            f"final_dfa_state={item['final_dfa_state']}"
        )

    output_path = Path(args.save_json)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()