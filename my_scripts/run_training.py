"""
# 使用默认配置文件
python ./my_scripts/run_training.py

# 使用自定义配置文件
python ./my_scripts/run_training.py --config ./my_config/your_custom_config.yaml
"""

import argparse
import yaml
import os
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Training script launcher')
    parser.add_argument('--config', type=str, default='./my_config/default_config.yaml',
                      help='Path to the configuration file')
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 生成时间戳
    current_time = datetime.now()
    date_str = current_time.strftime("%Y.%m.%d")
    time_str = current_time.strftime("%H.%M.%S")

    # 在构建命令之前添加
    name = config.get('name', '')
    task_name = config.get('task_name', '')

    
    # 构建命令
    cmd = (
        f"accelerate launch "
        f"--num_processes {config['num_processes']} "
        f"--num_cpu_threads_per_process {config['num_cpu_threads_per_process']} "
        f"train.py "
        f"--config-name={config['config_name']} "
        f"training.seed={config['training_seed']} "
        f"training.device={config['gpu_device']} "
        f"hydra.run.dir='data/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}{name}{task_name}' "
        f"{'task=' + config['task'] if 'task' in config else ''}"
    ).strip()
    
    # 执行命令
    print(f"Executing command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main() 