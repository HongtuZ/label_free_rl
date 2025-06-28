#!/bin/bash

# 定义 certain_beta 值数组
certain_betas=(0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)

# 定义可用 GPU 数组
gpus=(4 5 6 7)

# 配置路径
config_path="config/point-robot.yaml"

# 基线类型
baseline="classifier"

# 遍历 certain_beta 值并分配 GPU
for i in "${!certain_betas[@]}"; do
  certain_beta=${certain_betas[$i]}
  gpu=${gpus[$((i % ${#gpus[@]}))]} # 根据索引循环分配 GPU

  echo "Running with certain_beta=${certain_beta} on GPU=${gpu}"
  
  # 运行命令
  python run_certain.py --config_path "$config_path" --gpu "$gpu" --baseline "$baseline" --certain_beta "$certain_beta" &
done

# 等待所有后台任务完成
wait

echo "All tasks completed."