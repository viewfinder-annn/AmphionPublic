# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faulthandler

faulthandler.enable()

import os
import argparse
import json
import pyworld as pw
from multiprocessing import cpu_count
import random


from utils.util import load_config
from preprocessors.processor import preprocess_dataset, prepare_align
from preprocessors.metadata import cal_metadata
from processors import acoustic_extractor, content_extractor, data_augment

def preprocess(cfg, args):
    """Proprocess raw data of single or multiple datasets (in cfg.dataset)

    Args:
        cfg (dict): dictionary that stores configurations
        args (ArgumentParser): specify the configuration file and num_workers
    """
    # Specify the output root path to save the processed data
    output_path = cfg.preprocess.processed_dir
    os.makedirs(output_path, exist_ok=True)

    ## Split train and test sets
    for dataset in cfg.dataset:
        preprocess_dataset(
            dataset,
            cfg.dataset_path[dataset],
            output_path,
            cfg.preprocess,
            cfg.task_type,
            is_custom_dataset=dataset in cfg.use_custom_dataset,
        )

    # Data augmentation: create new wav files with pitch shift, formant shift, equalizer, time stretch
    try:
        assert isinstance(
            cfg.preprocess.data_augment, list
        ), "Please provide a list of datasets need to be augmented."
        if len(cfg.preprocess.data_augment) > 0:
            new_datasets_list = []
            for dataset in cfg.preprocess.data_augment:
                new_datasets = data_augment.augment_dataset(cfg, dataset)
                new_datasets_list.extend(new_datasets)
            cfg.dataset.extend(new_datasets_list)
            print("Augmentation datasets: ", cfg.dataset)
    except:
        print("No Data Augmentation.")

    for dataset in cfg.dataset:
        if not os.path.exists(os.path.join(output_path, dataset, cfg.preprocess.train_file)):
            # split the dataset into train, valid
            meta_file = os.path.join(output_path, dataset, "meta.json")
            meta_list = json.load(open(meta_file, "r"))
            with open(os.path.join(output_path, dataset, cfg.preprocess.train_file), "w") as f:
                json.dump(meta_list, f, indent=4, ensure_ascii=False)
            valid_list = random.sample(meta_list, 100)
            with open(os.path.join(output_path, dataset, cfg.preprocess.valid_file), "w") as f:
                json.dump(valid_list, f, indent=4, ensure_ascii=False)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="json files for configurations."
    )
    parser.add_argument("--num_workers", type=int, default=int(cpu_count()))
    parser.add_argument("--prepare_alignment", type=bool, default=False)

    args = parser.parse_args()
    cfg = load_config(args.config)

    preprocess(cfg, args)


if __name__ == "__main__":
    main()
