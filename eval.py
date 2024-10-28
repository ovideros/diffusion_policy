"""
使用方法:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# 为stdout和stderr使用行缓冲，确保实时输出日志
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)  # 指定检查点文件路径
@click.option('-o', '--output_dir', required=True)  # 指定输出目录
@click.option('-d', '--device', default='cuda:0')   # 指定运行设备，默认为CUDA:0
def main(checkpoint, output_dir, device):
    # 检查输出目录是否已存在，如果存在则询问是否覆盖
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载检查点
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 从工作空间获取策略
    policy = workspace.model
    # 如果配置中指定使用EMA模型，则使用EMA模型
    if hasattr(cfg.training, 'use_ema') and cfg.training.use_ema:
        policy = workspace.ema_model
    
    # 将策略移动到指定设备并设置为评估模式
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # 运行评估
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # 将日志转换为JSON格式并保存
    json_log = dict()
    for key, value in runner_log.items():
        # 对于wandb视频类型，保存其路径
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
