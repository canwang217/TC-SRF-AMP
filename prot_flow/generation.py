from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from config import create_config
from flow_matching_utils.flow_matching_holder import FlowMatchingRunner
from utils.setup_ddp import setup_ddp
from utils.util import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate peptide sequences from a trained flow checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to a flow checkpoint containing EMA weights.")
    parser.add_argument("--num-gen-texts", type=int, default=None, help="Number of sequences to generate.")
    parser.add_argument("--test-num", default=None, help="Optional suffix used by reflow data generation outputs.")
    parser.add_argument("--prefix", default="ProtFlow", help="Output filename/checkpoint prefix.")
    args = parser.parse_args()

    config = create_config()
    config.checkpoints_prefix = args.prefix
    config.project_name = args.prefix
    if args.num_gen_texts is not None:
        config.validation.num_gen_texts = args.num_gen_texts

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

    seed = config.seed + print_rank
    set_seed(seed)
    if print_rank == 0:
        print(config)

    flow_matching = FlowMatchingRunner(config, latent_mode=config.model.embeddings_type)
    flow_matching.test(args.ckpt, args.test_num)


if __name__ == "__main__":
    main()
