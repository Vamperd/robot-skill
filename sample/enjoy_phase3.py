import gymnasium as gym
from stable_baselines3 import PPO
import time
# 从你刚才的训练文件中导入我们写好的环境类
from sample.phase3_single_agent_rl import ProductMDPEnv

def watch_trained_agent():
    print("[SYSTEM] 正在加载训练好的大脑...")
    
    # 1. 实例化环境（开启图形渲染模式）
    env = ProductMDPEnv(render_mode="human")
    
    # 2. 加载数字大脑
    try:
        model = PPO.load("ppo_phase3_agent")
        print("[SYSTEM] 模型加载成功！开始无限循环播放，按窗口关闭按钮退出。")
    except FileNotFoundError:
        print("[ERROR] 找不到模型文件！请确保你已经先运行了训练脚本并保存了模型。")
        return

    # 3. 开启无限循环的死循环播放
    while True:
        obs, info = env.reset()
        done = False
        
        while not done:
            # deterministic=True 表示关闭探索随机性，采取神经网络认为的最优绝对动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 为了让你看清动作，稍微加一点延迟（如果在你的电脑上跑得太快的话）
            time.sleep(0.1) 
            
            done = terminated or truncated
            
        print("[SYSTEM] 任务链执行完毕，重置环境并再来一次！")
        time.sleep(1) # 胜利后停顿 1 秒再重置

if __name__ == "__main__":
    watch_trained_agent()