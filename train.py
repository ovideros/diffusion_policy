"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import datetime
from typing import Any

# 为stdout和stderr使用行缓冲
# 这确保了日志输出能够实时显示，对于长时间运行的训练过程很有用
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# 添加时间相关的解析器
def get_timestamp() -> str:
    return datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

def get_date() -> str:
    return datetime.datetime.now().strftime('%Y.%m.%d')

def get_time() -> str:
    return datetime.datetime.now().strftime('%H.%M.%S')

# 注册新的解析器
OmegaConf.register_new_resolver("timestamp", get_timestamp)
OmegaConf.register_new_resolver("date", get_date)
OmegaConf.register_new_resolver("time", get_time)
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # 立即解析配置，以确保所有${now:}解析器使用相同的时间
    # 这对于日志记录和实验复现很重要
    OmegaConf.resolve(cfg)

    # 根据配置中指定的类名获取相应的类
    cls = hydra.utils.get_class(cfg._target_)
    # 创建工作空间实例
    workspace: BaseWorkspace = cls(cfg)
    # 运行工作空间（开始训练或评估）
    workspace.run()

if __name__ == "__main__":
    main()
