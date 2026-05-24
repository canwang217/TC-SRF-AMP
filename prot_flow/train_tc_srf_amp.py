import os
import torch
import torch.distributed as dist

from flow_matching_utils.flow_matching_holder import FlowMatchingRunner
from utils.util import set_seed
from tc_srf_amp_config import create_tc_srf_amp_config
from utils.setup_ddp import setup_ddp


if __name__ == '__main__':
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(workspace_root, ".cache", "huggingface"))
    os.environ.setdefault("HF_HOME", os.path.join(workspace_root, ".cache", "huggingface"))

    config = create_tc_srf_amp_config()
    config.task.stage = "amp_finetune"
    config.checkpoints_prefix = "tc_srf_amp_finetune"

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    config.ddp = use_ddp

    if use_ddp:
        config.local_rank = setup_ddp()
        config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
        config.device = f"cuda:{dist.get_rank()}"
        print_rank = dist.get_rank()
    else:
        config.local_rank = 0
        config.training.batch_size_per_gpu = config.training.batch_size
        config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print_rank = 0

    config.project_name = 'TC-SRF-AMP'

    seed = config.seed
    set_seed(seed)
    if print_rank == 0:
        print(config)

    flow_matching = FlowMatchingRunner(config, latent_mode=config.model.embeddings_type)

    seed = config.seed + print_rank
    set_seed(seed)
    flow_matching.train(
        project_name=config.project_name,
        experiment_name=config.checkpoints_prefix
    )
