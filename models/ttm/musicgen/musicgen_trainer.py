# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json5
import os
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base.new_trainer import BaseTrainer

import modules.audiocraft.solvers.builders as builders
import modules.audiocraft.models as models
from modules.audiocraft.solvers.compression import CompressionSolver
from modules.audiocraft.data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from modules.audiocraft.modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition
from modules.audiocraft.utils.autocast import TorchAutocast
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
        self.autocast_dtype = {
            'float16': torch.float16, 'bfloat16': torch.bfloat16
        }[self.cfg_omega.autocast_dtype]
        self.autocast = TorchAutocast(enabled=self.cfg_omega.autocast, device_type=self.device, dtype=self.autocast_dtype)
        # self.scaler = torch.cuda.amp.GradScaler()

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
        return dataloader_dict['train'], dataloader_dict['train']
    
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

    # adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/solvers/musicgen.py
    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ):
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def _prepare_tokens_and_attributes(self, batch):
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        # TODO: cached dataset support
        audio, infos = batch
        audio = audio.to(self.device)
        audio_tokens = None
        assert audio.size(0) == len(infos), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(infos)})"
        )
        
        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos]
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        if audio_tokens is None:
            with torch.no_grad():
                audio_tokens, scale = self.compression_model.encode(audio)
                assert scale is None, "Scaled compression model not supported with LM."

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.cfg.tokens.padding_with_special_token:
            audio_tokens = audio_tokens.clone()
            padding_mask = padding_mask.clone()
            token_sample_rate = self.compression_model.frame_rate
            B, K, T_s = audio_tokens.shape
            for i in range(B):
                n_samples = infos[i].n_frames
                audio_sample_rate = infos[i].sample_rate
                # take the last token generated from actual audio frames (non-padded audio)
                valid_tokens = math.floor(float(n_samples) / audio_sample_rate * token_sample_rate)
                audio_tokens[i, :, valid_tokens:] = self.model.special_token_id
                padding_mask[i, :, valid_tokens:] = 0

        return condition_tensors, audio_tokens, padding_mask

    def _forward_step(self, batch):
        r"""Forward step for the model. This function is called in ``_train_step`` and ``_valid_step`` function."""
        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(batch)
        
        with self.autocast:
            model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
            logits = model_output.logits
            mask = padding_mask & model_output.mask
            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
            loss = ce
        
        # if self.scaler is not None:
        #     loss = self.scaler.scale(loss)
        
        # if self.is_training:
        #     if self.scaler is not None:
        #         loss = self.scaler.scale(loss)
        #     loss.backward()

        #     if self.scaler is not None:
        #         self.scaler.unscale_(self.optimizer)
        #     if self.cfg.optim.max_norm:
        #         if self.cfg.fsdp.use:
        #             grad_norm = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
        #         else:
        #             grad_norm = torch.nn.utils.clip_grad_norm_(
        #                 self.model.parameters(), self.cfg.optim.max_norm
        #             )
        #     if self.scaler is None:
        #         self.optimizer.step()
        #     else:
        #         self.scaler.step(self.optimizer)
        #         self.scaler.update()
        #     if self.lr_scheduler:
        #         self.lr_scheduler.step()
        #     self.optimizer.zero_grad()
        #     self.deadlock_detect.update('optim')
        #     if self.scaler is not None:
        #         scale = self.scaler.get_scale()
        #         grad_scale = scale
        #     if not loss.isfinite().all():
        #         raise RuntimeError("Model probably diverged.")

        # metrics['ce'] = ce
        # metrics['ppl'] = torch.exp(ce)
        # for k, ce_q in enumerate(ce_per_codebook):
        #     metrics[f'ce_q{k + 1}'] = ce_q
        #     metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)

        return loss