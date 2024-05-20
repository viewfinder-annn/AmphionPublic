# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/solvers/musicgen.py

import json5
import json
import os
import shutil
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import logging

from models.base.new_trainer import BaseTrainer
from models.ttm.sing2song.sing2song_dataset import Sing2SongDataset, Sing2SongCollator
import modules.audiocraft.solvers.builders as builders
import modules.audiocraft.models as models
from modules.audiocraft.solvers.compression import CompressionSolver
from modules.audiocraft.data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from modules.audiocraft.modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, T5Conditioner
from modules.audiocraft.utils.autocast import TorchAutocast
import omegaconf
import torchaudio

logging.basicConfig(level=logging.INFO)

import datetime
torch.manual_seed(datetime.datetime.now().microsecond)

class Sing2SongTrainer(BaseTrainer):

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
        
        # TODO: resume
        self.checkpoints_path_step = [
            [] for _ in range(len(self.cfg.train.save_checkpoint_stride_step))
        ]
        self.keep_last_step = [
            i if i > 0 else float("inf") for i in self.cfg.train.keep_last_step
        ]

    def _build_dataset(self):
        return Sing2SongDataset, Sing2SongCollator

    ### Following are methods only for TTM tasks ###
    def _build_dataloader(self):
        r"""Build the dataloader for training. This function is called in ``__init__`` function."""
        
        if self.cfg.train.use_audiocraft_dataset:
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
        else:
            Dataset, Collator = self._build_dataset()
            datasets_list = []
            for dataset in self.cfg.dataset:
                subdataset = Dataset(self.cfg, dataset, is_valid=False)
                datasets_list.append(subdataset)
            train_dataset = ConcatDataset(datasets_list)
            train_collate = Collator(self.cfg)
            train_dataloader = DataLoader(train_dataset, collate_fn=train_collate, batch_size=self.cfg.train.batch_size, shuffle=True, pin_memory=True)

            datasets_list = []
            for dataset in self.cfg.dataset:
                subdataset = Dataset(self.cfg, dataset, is_valid=True)
                datasets_list.append(subdataset)
            valid_dataset = ConcatDataset(datasets_list)
            valid_collate = Collator(self.cfg)
            valid_dataloader = DataLoader(valid_dataset, collate_fn=valid_collate, batch_size=self.cfg.train.batch_size, shuffle=False, pin_memory=True)
            
            # DEBUG
            debug_sample_dir = f"{self.exp_dir}/debug_train_sample"
            shutil.rmtree(debug_sample_dir, ignore_errors=True)
            os.makedirs(debug_sample_dir, exist_ok=True)
            debug_infos = []
            for i in range(8):
                self_wav, ref_wav, infos = train_dataset[i]
                # print(wavs.shape)
                # print(infos)
                # print(infos.to_condition_attributes())
                torchaudio.save(f"{debug_sample_dir}/{i}_self.wav", self_wav, self.cfg_omega.sample_rate)
                torchaudio.save(f"{debug_sample_dir}/{i}_ref.wav", ref_wav, self.cfg_omega.sample_rate)
                debug_infos.append({
                    "dataset": "debug",
                    "action": infos.text["action"] if "action" in infos.text else "add",
                    "category": infos.text["category"] if "category" in infos.text else "accom",
                    "ref_wav": f"{debug_sample_dir}/{i}_ref.wav",
                    "self_wav": f"{debug_sample_dir}/{i}_self.wav",
                })
            with open(f"{debug_sample_dir}/info.json", "w") as f:
                json.dump(debug_infos, f)
            return train_dataloader, valid_dataloader
            
    
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
        else:
            scheduler = None
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
        audio, prompt, infos = batch
        audio = audio.to(self.device)
        prompt = prompt.to(self.device)
        audio_tokens = None
        assert audio.size(2) == prompt.size(2), "Audio and prompt should have the same length."
        assert audio.size(0) == len(infos), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(infos)})"
        )
        
        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos]
        # print(attributes)
        attributes = self.accelerator.unwrap_model(self.model).cfg_dropout(attributes)
        attributes = self.accelerator.unwrap_model(self.model).att_dropout(attributes)
        tokenized = self.accelerator.unwrap_model(self.model).condition_provider.tokenize(attributes)
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
                prompt_tokens, _ = self.compression_model.encode(prompt)
                assert prompt_tokens.shape == audio_tokens.shape, "Prompt and audio tokens should have the same shape."
                audio_tokens = torch.cat([prompt_tokens, audio_tokens], dim=2)

        with self.autocast:
            condition_tensors = self.accelerator.unwrap_model(self.model).condition_provider(tokenized)
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
            
            if self.accelerator.is_main_process:
                save_checkpoint = False
                hit_dix = []
                for i, num in enumerate(self.cfg.train.save_checkpoint_stride_step):
                    if self.step % num == 0:
                        save_checkpoint = True
                        hit_dix.append(i)

            self.accelerator.wait_for_everyone()
            train_loss = epoch_sum_loss / epoch_step if epoch_step > 0 else 0.0
            if self.accelerator.is_main_process and save_checkpoint:
                if torch.isnan(train_loss):
                    raise ValueError("NaN loss encountered during training, aborting.")
                path = os.path.join(
                    self.checkpoint_dir,
                    "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_loss
                    ),
                )
                self.tmp_checkpoint_save_path = path
                self.accelerator.save_state(path)
                print(f"save checkpoint in {path}")
                json.dump(
                    self.checkpoints_path_step,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                self._save_auxiliary_states()

                # Remove old checkpoints
                to_remove = []
                for idx in hit_dix:
                    self.checkpoints_path_step[idx].append(path)
                    while len(self.checkpoints_path_step[idx]) > self.keep_last_step[idx]:
                        to_remove.append((idx, self.checkpoints_path_step[idx].pop(0)))

                # Search conflicts
                total = set()
                for i in self.checkpoints_path_step:
                    total |= set(i)
                do_remove = set()
                for idx, path in to_remove[::-1]:
                    if path in total:
                        self.checkpoints_path_step[idx].insert(0, path)
                    else:
                        do_remove.add(path)
                
                # Remove old checkpoints
                for path in do_remove:
                    shutil.rmtree(path, ignore_errors=True)
                    self.logger.debug(f"Remove old checkpoint: {path}")
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
            model_output = self.accelerator.unwrap_model(self.model).compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
            logits = model_output.logits
            # print("logits", logits.shape)
            mask = model_output.mask
            # remove the prompt tokens from the logits, leave the second half [B, K, 2T, num_cards] -> [B, K, T, num_cards]
            logits = logits[:, :, logits.shape[2] // 2:]
            # print("logits", logits.shape)
            mask = padding_mask & model_output.mask
            # print("mask", mask.shape)
            mask = mask[:, :, mask.shape[-1] // 2:]
            # print("mask", mask.shape)
            # print("audio_tokens", audio_tokens.shape)
            audio_tokens = audio_tokens[:, :, audio_tokens.shape[-1] // 2:]
            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
            loss = ce

        return loss