import numpy as np
import matplotlib.pyplot as plt
import os

# 指定 evaluations.npz 的所在路径（若找不到文件请修改此处）
LOG_PATH = "./generalization_eval__vel_punishment_logs/evaluations.npz"

def plot_comprehensive_curves(log_path):
    if not os.path.exists(log_path):
        print(f"Error: 找不到文件 {log_path}。请确认路径是否正确。")
        return

    # 1. 加载 npz 文件数据
    print(f"正在读取 {log_path}...")
    data = np.load(log_path)
    
    # 提取评估数据
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']
    
    # 2. 计算各指标均值和标准差 (axis=1 是对多次评价求统计量)
    # Episode Reward
    reward_mean = np.mean(results, axis=1)
    reward_std = np.std(results, axis=1)
    
    # Episode Length
    length_mean = np.mean(ep_lengths, axis=1)
    length_std = np.std(ep_lengths, axis=1)
    
    # Success Rate (如果包含在返回文件中)
    has_success = 'successes' in data
    if has_success:
        successes = data['successes']
        success_mean = np.mean(successes, axis=1)
        success_std = np.std(successes, axis=1)
    else:
        print("Warning: evaluations.npz 中未找到 'successes' 数组。")

    # 3. 创建 1行3列 的子图表，尺寸设为横向宽屏 18x5
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 左子图：Episode Reward 曲线 (使用深蓝色) ---
    ax1 = axes[0]
    ax1.plot(timesteps, reward_mean, color='#1f77b4', linewidth=2.5, label='Mean Reward')
    ax1.fill_between(timesteps, 
                     reward_mean - reward_std, 
                     reward_mean + reward_std, 
                     color='#1f77b4', alpha=0.2)
    ax1.set_title('Episode Reward over Timesteps', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=10)

    # --- 中子图：Success Rate 曲线 (使用绿色) ---
    ax2 = axes[1]
    if has_success:
        ax2.plot(timesteps, success_mean, color='#2ca02c', linewidth=2.5, label='Mean Success Rate')
        ax2.fill_between(timesteps, 
                         success_mean - success_std, 
                         success_mean + success_std, 
                         color='#2ca02c', alpha=0.2)
        ax2.set_title('Success Rate over Timesteps', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timesteps', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.set_ylim(-0.05, 1.05) # 固定在 0 ~ 1.0 之间
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='lower right', fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No Success Rate Data", ha='center', va='center', fontsize=12)
        ax2.set_title('Success Rate over Timesteps', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timesteps', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)

    # --- 右子图：Episode Length 曲线 (使用橙色) ---
    ax3 = axes[2]
    ax3.plot(timesteps, length_mean, color='#ff7f0e', linewidth=2.5, label='Mean Episode Length')
    ax3.fill_between(timesteps, 
                     length_mean - length_std, 
                     length_mean + length_std, 
                     color='#ff7f0e', alpha=0.2)
    ax3.set_title('Episode Length over Timesteps', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Timesteps', fontsize=12)
    ax3.set_ylabel('Steps (Lower is faster/better)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right', fontsize=10)

    # 4. 调整布局以保证不重叠并保存图像
    plt.tight_layout()
    save_filename = "comprehensive_training_curves.png"
    plt.savefig(save_filename, dpi=300)
    print(f"图表已成功绘制并保存至当前目录下的: {save_filename}")
    
    # 5. 阻断弹窗展示图片
    plt.show()


if __name__ == "__main__":
    plot_comprehensive_curves(LOG_PATH)
