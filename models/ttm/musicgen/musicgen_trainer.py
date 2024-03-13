# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/solvers/musicgen.py

import json5
import os
import shutil
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from models.base.new_trainer import BaseTrainer

import modules.audiocraft.solvers.builders as builders
import modules.audiocraft.models as models
from modules.audiocraft.solvers.compression import CompressionSolver
from modules.audiocraft.data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from modules.audiocraft.modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, T5Conditioner
from modules.audiocraft.utils.autocast import TorchAutocast
import omegaconf
import torchaudio

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
        self.cfg_omega = omegaconf.OmegaConf.create(json5.load(open(cfg_path))['audiocraft'])
        self.device = self.cfg_omega.device
        self.autocast_dtype = {
            'float16': torch.float16, 'bfloat16': torch.bfloat16
        }[self.cfg_omega.autocast_dtype]

        self._init_accelerator()

        # Super init
        BaseTrainer.__init__(self, args, cfg)
        
        self.autocast = TorchAutocast(enabled=self.cfg_omega.autocast, device_type=self.device, dtype=self.autocast_dtype)
        self.scaler = None
        need_scaler = self.cfg_omega.autocast and self.autocast_dtype is torch.float16
        if need_scaler:
            print("Using GradScaler for autocast with float16")
            self.accelerator.scaler = torch.cuda.amp.GradScaler()

        # Only for TTM tasks
        self.task_type = "TTM"
        self.logger.info("Task type: {}".format(self.task_type))

    ### Following are methods only for TTM tasks ###
    def _build_dataloader(self):
        r"""Build the dataloader for training. This function is called in ``__init__`` function."""
        dataloader_dict = builders.get_audio_datasets(self.cfg_omega, dataset_type=self.DATASET_TYPE)
        
        ### DEBUG
        dataloader_train = dataloader_dict['train']
        debug_sample_dir = f"{self.exp_dir}/debug_train_sample"
        shutil.rmtree(debug_sample_dir, ignore_errors=True)
        os.makedirs(debug_sample_dir, exist_ok=True)
        with open(f"{debug_sample_dir}/info", "w") as f:
            for data in dataloader_train:
                wavs, infos = data
                # print(wavs.shape)
                for i in range(wavs.shape[0]):
                    # print(wavs[i])
                    # print(infos[i])
                    f.write(f"{infos[i]}\n")
                    # print(infos[i].to_condition_attributes())
                    torchaudio.save(f"{debug_sample_dir}/{infos[i].description[:100].replace('/', '-')}.wav", wavs[i], self.cfg_omega.sample_rate)
                break
                
        
        return dataloader_dict['train'], dataloader_dict['valid']
    
    def _build_model(self):
        r"""Build the model for training. This function is called in ``__init__`` function."""
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
        ### Conditioner info
        for k, v in self.model.condition_provider.conditioners.items():
            self.logger.info(f"Conditioner {k}: {v}")
            if isinstance(v, T5Conditioner):
                if v.name == "t5-base":
                    self.logger.info("Using T5-base as the condition provider. (~110M parameters)")

        # TODO: distributed training
        # if self.cfg_omega.fsdp.use:
        #     assert not self.cfg_omega.autocast, "Cannot use autocast with fsdp"
        #     self.model = self.wrap_with_fsdp(self.model)
        
        # ema
        # self.register_ema('model')
        return self.model
    
    # def _build_optimizer(self):
    #     r"""Build the optimizer for training. This function is called in ``__init__`` function."""
    #     return builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg_omega.optim)
    
    def _build_scheduler(self):
        if self.cfg.train.scheduler.lower() == "cosine":
            from diffusers.optimization import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.cfg.train.warmup_steps
                * self.accelerator.num_processes,
                num_training_steps=self.cfg.train.total_training_steps
                * self.accelerator.num_processes,
            )
        return scheduler

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
        # print(attributes)
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)
        # for k, v in tokenized.items():
        #     print(k)
        #     if isinstance(v, dict):
        #         for key, value in v.items():
        #             print(f"{key}: {value.shape}")
        #     else:
        #         print(v)

        if audio_tokens is None:
            with torch.no_grad():
                audio_tokens, scale = self.compression_model.encode(audio)
                assert scale is None, "Scaled compression model not supported with LM."

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)
            # for k, v in condition_tensors.items():
            #     print(k)
            #     print(v[0].shape, v[1].shape)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.cfg_omega.tokens.padding_with_special_token:
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

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.train()
        epoch_sum_loss: float = 0.0
        epoch_step: int = 0
        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                loss = self._train_step(batch)
                # if self.scaler is not None:
                #     loss = self.scaler.scale(loss)

                self.accelerator.backward(loss)
                
                # if self.scaler is not None:
                #     self.scaler.unscale_(self.optimizer)
                
                if self.cfg.train.grad_clip_thresh:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_thresh)
                    
                self.optimizer.step()
                # if self.scaler is not None:
                #     self.optimizer.step()
                # else:
                #     self.scaler.step(self.optimizer)
                #     self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            self.batch_count += 1

            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += loss
                self.accelerator.log(
                    {
                        "Step/Train Loss": loss,
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                self.step += 1
                epoch_step += 1

        self.accelerator.wait_for_everyone()
        return (
            epoch_sum_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )

    def _forward_step(self, batch):
        r"""Forward step for the model. This function is called in ``_train_step`` and ``_valid_step`` function."""
        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(batch)
        
        with self.autocast:
            model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
            logits = model_output.logits
            mask = padding_mask & model_output.mask
            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
            loss = ce

        return loss