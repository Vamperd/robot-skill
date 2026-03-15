import numpy as np
import matplotlib.pyplot as plt
import os

# 指定 evaluations.npz 的所在路径（若找不到文件请修改此处）
LOG_PATH = "./generalization_eval__vel_punishment_logs/evaluations.npz"

def plot_curves(log_path):
    if not os.path.exists(log_path):
        print(f"Error: 找不到文件 {log_path}。请确认路径是否正确。")
        return

    # 1. 加载 npz 文件数据
    print(f"正在读取 {log_path}...")
    data = np.load(log_path)
    
    # EvalCallback 默认保存三个主要的 key: timesteps, results, ep_lengths
    # 如果环境 step() info 中包含了 is_success 的 key，则还会包含 successes
    timesteps = data['timesteps']
    results = data['results']
    
    # 2. 计算奖励（Reward）的均值和标准差 (axis=1 代表沿着评估的 episode 求均值)
    reward_mean = np.mean(results, axis=1)
    reward_std = np.std(results, axis=1)
    
    # 计算成功率（Success Rate）的均值和标准差
    has_success = 'successes' in data
    if has_success:
        successes = data['successes']
        success_mean = np.mean(successes, axis=1)
        success_std = np.std(successes, axis=1)
    else:
        print("Warning: evaluations.npz 中未找到 'successes' 数组，无法绘制出准确的成功率曲线。")

    # 3. 创建 1行2列 的图表，尺寸为 12x5
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左子图：Episode Reward 曲线 ---
    ax1 = axes[0]
    ax1.plot(timesteps, reward_mean, color='#1f77b4', linewidth=2, label='Mean Reward')
    ax1.fill_between(timesteps, 
                     reward_mean - reward_std, 
                     reward_mean + reward_std, 
                     color='#1f77b4', alpha=0.2, label='± 1 Std Dev')
    
    ax1.set_title('Episode Reward over Timesteps', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=10)

    # --- 右子图：Success Rate 曲线 ---
    ax2 = axes[1]
    if has_success:
        ax2.plot(timesteps, success_mean, color='#2ca02c', linewidth=2, label='Mean Success Rate')
        ax2.fill_between(timesteps, 
                         success_mean - success_std, 
                         success_mean + success_std, 
                         color='#2ca02c', alpha=0.2)
        ax2.set_title('Success Rate over Timesteps', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timesteps', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=12)
        # 固定成功率的 Y 轴范围在 0 到 1 之间
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='lower right', fontsize=10)
    else:
        # 兼容处理：如果没有 success 数据，显示提示信息
        ax2.text(0.5, 0.5, "No Success Rate Data", ha='center', va='center', fontsize=12)
        ax2.set_title('Success Rate over Timesteps', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timesteps', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)

    # 4. 调整布局，保存高清图并展示
    plt.tight_layout()
    save_filename = "training_curves.png"
    plt.savefig(save_filename, dpi=300)
    print(f"图表已成功绘制并保存至当前目录下的: {save_filename}")
    
    plt.show()

if __name__ == "__main__":
    plot_curves(LOG_PATH)
