"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import os
import torch.distributed as dist
import hydra
import pathlib
from omegaconf import OmegaConf
import torch

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # 初始化分布式环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    OmegaConf.resolve(cfg)
    
    # 将rank和world_size添加到配置中
    cfg.rank = rank
    cfg.world_size = world_size
    cfg.local_rank = local_rank

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()