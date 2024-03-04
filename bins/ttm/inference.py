# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser
import os

from models.ttm.musicgen.musicgen_inference import MusicGenInference
from utils.util import save_config, load_model_config, load_config
import numpy as np
import torch


def build_inference(args, cfg, cfg_path=None):
    supported_inference = {
        "MusicGen": MusicGenInference,
    }

    inference_class = supported_inference[cfg.model_type]
    inference = inference_class(args, cfg, cfg_path=cfg_path)
    return inference


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON/YAML file for configurations.",
    )
    parser.add_argument(
        "--text",
        help="Text to be synthesized",
        type=str,
        default=None
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Text file to be synthesized",
    )
    parser.add_argument(
        "--infer_expt_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir for saving generated results",
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    return parser


def main():
    # Parse arguments
    args = build_parser().parse_args()
    # args, infer_type = formulate_parser(args)

    # Parse config
    cfg = load_config(args.config)
    if torch.cuda.is_available():
        args.local_rank = torch.device("cuda")
    else:
        args.local_rank = torch.device("cpu")
    print("args: ", args)
    if args.text == "" and args.text_file == "":
        raise ValueError("Please provide either --text or --text_file")

    # Build inference
    inferencer = build_inference(args, cfg, cfg_path=args.config)

    # Run inference
    inferencer.inference()


if __name__ == "__main__":
    main()
