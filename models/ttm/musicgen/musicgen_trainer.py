# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json5
import os

import torch
import torch.nn as nn

from models.base.new_trainer import BaseTrainer

import modules.audiocraft.solvers.builders as builders
import modules.audiocraft.models as models
from modules.audiocraft.solvers.compression import CompressionSolver
import omegaconf

class MusicGenTrainer(BaseTrainer):
    r"""The base trainer for all MusicGen models. It inherits from BaseTrainer and implements
    """

    DATASET_TYPE = builders.DatasetType.MUSIC

    def __init__(self, args=None, cfg=None, cfg_path=None):
        self.args = args
        self.cfg = cfg
        
        # print(type(cfg))
        # exit()
        # for audiocraft compatibility
        self.cfg_omega = omegaconf.OmegaConf.create(json5.load(open(cfg_path)))
        self.device = cfg.device

        self._init_accelerator()

        # Super init
        BaseTrainer.__init__(self, args, cfg)

        # Only for TTM tasks
        self.task_type = "TTM"
        self.logger.info("Task type: {}".format(self.task_type))

    ### Following are methods only for TTM tasks ###
    def _build_dataloader(self):
        r"""Build the dataloader for training. This function is called in ``__init__`` function."""
        dataloader_dict = builders.get_audio_datasets(self.cfg_omega, dataset_type=self.DATASET_TYPE)
        return dataloader_dict['train'], dataloader_dict['valid']
    
    def _build_model(self):
        r"""Build the model for training. This function is called in ``__init__`` function."""
        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg_omega, self.cfg_omega.compression_model_checkpoint, device=self.device)
        
        # we can potentially not use all quantizers with which the EnCodec model was trained
        # (e.g. we trained the model with quantizers dropout)
        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg_omega, self.cfg_omega.compression_model_checkpoint, device=self.device)
        assert self.compression_model.sample_rate == self.cfg_omega.sample_rate, (
            f"Compression model sample rate is {self.compression_model.sample_rate} but "
            f"Solver sample rate is {self.cfg_omega.sample_rate}."
            )
        # ensure we have matching configuration between LM and compression model
        assert self.cfg_omega.transformer_lm.card == self.compression_model.cardinality, (
            "Cardinalities of the LM and compression model don't match: ",
            f"LM cardinality is {self.cfg_omega.transformer_lm.card} vs ",
            f"compression model cardinality is {self.compression_model.cardinality}"
        )
        assert self.cfg_omega.transformer_lm.n_q == self.compression_model.num_codebooks, (
            "Numbers of codebooks of the LM and compression models don't match: ",
            f"LM number of codebooks is {self.cfg_omega.transformer_lm.n_q} vs ",
            f"compression model numer of codebooks is {self.compression_model.num_codebooks}"
        )
        self.logger.info("Compression model has %d codebooks with %d cardinality, and a framerate of %d",
                            self.compression_model.num_codebooks, self.compression_model.cardinality,
                            self.compression_model.frame_rate)

        # instantiate LM model
        self.model: models.LMModel = models.builders.get_lm_model(self.cfg_omega).to(self.device)
        if self.cfg_omega.fsdp.use:
            assert not self.cfg_omega.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model)
        
        # ema
        # self.register_ema('model')
        return self.model
    
    def _build_optimizer(self):
        r"""Build the optimizer for training. This function is called in ``__init__`` function."""
        return builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg_omega.optim)
    
    def _build_scheduler(self):
        self.train_updates_per_epoch = len(self.train_dataloader) if self.train_dataloader else 0
        if self.cfg_omega.optim.updates_per_epoch:
            self.train_updates_per_epoch = self.cfg_omega.optim.updates_per_epoch
        self.total_updates = self.train_updates_per_epoch * self.cfg_omega.optim.epochs
        return builders.get_lr_scheduler(self.optimizer, self.cfg_omega.schedule, self.total_updates)

    def _forward_step(self, batch):
        raise NotImplementedError