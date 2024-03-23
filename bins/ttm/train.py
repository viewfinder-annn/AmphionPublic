# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from models.ttm.musicgen.musicgen_trainer import MusicGenTrainer
from models.ttm.latentdiffusion.autoencoder_trainer import AutoencoderKLTrainer
from models.ttm.latentdiffusion.audioldm_trainer import AudioLDMTrainer
from utils.util import load_config


def build_trainer(args, cfg, cfg_path=None):
    supported_trainer = {
        "MusicGen": MusicGenTrainer,
        "AutoencoderKL": AutoencoderKLTrainer,
        "AudioLDM": AudioLDMTrainer
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg, cfg_path=cfg_path)
    return trainer


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If specified, to resume from the existing checkpoint.",
    )
    parser.add_argument(
        "--resume_from_ckpt_path",
        type=str,
        default="",
        help="The specific checkpoint path that you want to resume from.",
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="",
        help="`resume` for loading all the things (including model weights, optimizer, scheduler, and random states). `finetune` for loading only the model weights",
    )

    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    args = parser.parse_args()
    # Model saving dir
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

    # Build trainer
    trainer = build_trainer(args, cfg, cfg_path=args.config)

    trainer.train_loop()


if __name__ == "__main__":
    main()
