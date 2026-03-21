import os
import pickle
import time
from scenario_generator import generate_random_scenario

def build_maps(num_maps=100, save_dir="offline_maps"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"开始生成 {num_maps} 张具有【5种构型】和【协同锁】的随机地图...")
    start_time = time.time()
    
    for i in range(1, num_maps + 1):
        print(f"正在生成 map_{i}.pkl ...")
        # 调用生成器（内部自带了 A* 连通性验证的拒绝采样）
        scenario_config = generate_random_scenario()
        
        # 序列化保存到硬盘
        file_path = os.path.join(save_dir, f"map_{i}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(scenario_config, f)
            
    total_time = time.time() - start_time
    print(f"\n✅ 成功！所有 {num_maps} 张离线地图生成完毕！")
    print(f"总耗时: {total_time:.2f} 秒，平均每张图耗时: {total_time/num_maps:.3f} 秒。")

if __name__ == "__main__":
    # 生成全新的 100 张地图，覆盖旧版本
    build_maps(num_maps=100)