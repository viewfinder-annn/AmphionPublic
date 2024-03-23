# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from models.base.new_trainer import BaseTrainer
from models.ttm.latentdiffusion.autoencoder_dataset import (
    AutoencoderKLDataset,
    AutoencoderKLCollator,
)
from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.autoencoder.autoencoder_loss import AutoencoderLossWithDiscriminator
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

import os
import json
import shutil
from tqdm import tqdm
import json5


class AutoencoderKLTrainer(BaseTrainer):
    # TODO: remove cfg_path
    def __init__(self, args, cfg, cfg_path=None):
        self.args = args
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        BaseTrainer.__init__(self, args, cfg)
        # Only for TTM tasks
        self.task_type = "TTM"
        self.logger.info("Task type: {}".format(self.task_type))

    def _build_dataset(self):
        return AutoencoderKLDataset, AutoencoderKLCollator

    def _build_optimizer(self):
        opt_ae = torch.optim.AdamW(self.model.parameters(), **self.cfg.train.adam)
        opt_disc = torch.optim.AdamW(
            self.criterion.discriminator.parameters(), **self.cfg.train.adam
        )
        optimizer = {"opt_ae": opt_ae, "opt_disc": opt_disc}
        return optimizer

    def _build_dataloader(self):
        Dataset, Collator = self._build_dataset()
        # build dataset instance for each dataset and combine them by ConcatDataset
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=False)
            datasets_list.append(subdataset)
        train_dataset = ConcatDataset(datasets_list)

        train_collate = Collator(self.cfg)
        # DEBUG
        


        # use batch_sampler argument instead of (sampler, shuffle, drop_last, batch_size)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_collate,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
        )
        if not self.cfg.train.ddp or self.args.local_rank == 0:
            datasets_list = []
            for dataset in self.cfg.dataset:
                subdataset = Dataset(self.cfg, dataset, is_valid=True)
                datasets_list.append(subdataset)
            valid_dataset = ConcatDataset(datasets_list)
            valid_collate = Collator(self.cfg)

            valid_loader = DataLoader(
                valid_dataset,
                collate_fn=valid_collate,
                batch_size=self.cfg.train.batch_size,
            )
        else:
            raise NotImplementedError("DDP is not supported yet.")
            # valid_loader = None
        return train_loader, valid_loader

    # TODO: check it...
    def _build_scheduler(self):
        return None
        # return ReduceLROnPlateau(self.optimizer["opt_ae"], **self.cfg.train.lronPlateau)

    def _build_criterion(self):
        return self.criterion

    # def get_state_dict(self):
    #     if self.scheduler != None:
    #         state_dict = {
    #             "model": self.model.state_dict(),
    #             "optimizer_ae": self.optimizer["opt_ae"].state_dict(),
    #             "optimizer_disc": self.optimizer["opt_disc"].state_dict(),
    #             "scheduler": self.scheduler.state_dict(),
    #             "step": self.step,
    #             "epoch": self.epoch,
    #             "batch_size": self.cfg.train.batch_size,
    #         }
    #     else:
    #         state_dict = {
    #             "model": self.model.state_dict(),
    #             "optimizer_ae": self.optimizer["opt_ae"].state_dict(),
    #             "optimizer_disc": self.optimizer["opt_disc"].state_dict(),
    #             "step": self.step,
    #             "epoch": self.epoch,
    #             "batch_size": self.cfg.train.batch_size,
    #         }
    #     return state_dict

    # def load_model(self, checkpoint):
    #     self.step = checkpoint["step"]
    #     self.epoch = checkpoint["epoch"]

    #     self.model.load_state_dict(checkpoint["model"])
    #     self.optimizer["opt_ae"].load_state_dict(checkpoint["optimizer_ae"])
    #     self.optimizer["opt_disc"].load_state_dict(checkpoint["optimizer_disc"])
    #     if self.scheduler != None:
    #         self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _build_model(self):
        self.model = AutoencoderKL(self.cfg.model.autoencoderkl)
        # need to build self.criterion right after model since _build_optimizer() needs self.criterion
        self.criterion = AutoencoderLossWithDiscriminator(self.cfg.model.loss)
        self.criterion = self.criterion.cuda()
        return self.model

    ### THIS IS MAIN ENTRY ###
    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self.__dump_cfg(self.config_save_path)
        self.model.train()
        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            self.logger.info("\n")
            self.logger.info("-" * 32)
            self.logger.info("Epoch {}: ".format(self.epoch))

            ### TODO: change the return values of _train_epoch() to a loss dict, or (total_loss, loss_dict)
            ### It's inconvenient for the model with multiple losses
            # Do training & validating epoch
            train_loss = self._train_epoch()
            self.logger.info("  |- Train/Loss: {:.6f}".format(train_loss))
            valid_loss = self._valid_epoch()
            self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_loss))
            self.accelerator.log(
                {"Epoch/Train Loss": train_loss, "Epoch/Valid Loss": valid_loss},
                step=self.epoch,
            )

            self.accelerator.wait_for_everyone()
            # TODO: what is scheduler?
            # self.scheduler.step(valid_loss)  # FIXME: use epoch track correct?

            # Check if hit save_checkpoint_stride and run_eval
            run_eval = False
            if self.accelerator.is_main_process:
                save_checkpoint = False
                hit_dix = []
                for i, num in enumerate(self.save_checkpoint_stride):
                    if self.epoch % num == 0:
                        save_checkpoint = True
                        hit_dix.append(i)
                        run_eval |= self.run_eval[i]

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and save_checkpoint:
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
                    self.checkpoints_path,
                    open(os.path.join(path, "ckpts.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
                self._save_auxiliary_states()

                # Remove old checkpoints
                to_remove = []
                for idx in hit_dix:
                    self.checkpoints_path[idx].append(path)
                    while len(self.checkpoints_path[idx]) > self.keep_last[idx]:
                        to_remove.append((idx, self.checkpoints_path[idx].pop(0)))

                # Search conflicts
                total = set()
                for i in self.checkpoints_path:
                    total |= set(i)
                do_remove = set()
                for idx, path in to_remove[::-1]:
                    if path in total:
                        self.checkpoints_path[idx].insert(0, path)
                    else:
                        do_remove.add(path)

                # Remove old checkpoints
                for path in do_remove:
                    shutil.rmtree(path, ignore_errors=True)
                    self.logger.debug(f"Remove old checkpoint: {path}")

            self.accelerator.wait_for_everyone()
            if run_eval:
                # TODO: run evaluation
                pass

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, valid_loss
                    ),
                )
            )
            self._save_auxiliary_states()

        self.accelerator.end_training()

    ### Following are methods that can be used directly in child classes ###
    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.train()
        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
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
                # adapted from models/svc/vits/vits_trainer.py
                train_losses, training_stats, total_loss, lr = self._train_step(batch)
            self.batch_count += 1

            # Update info for each step
            # TODO: step means BP counts or batch counts?
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value
                self.accelerator.log(
                    {
                        **{f"Train/{key}": value for key, value in train_losses.items()},
                        "Step/Learning Rate": lr,
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

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        self.model.eval()
        epoch_sum_loss = 0.0
        for batch in tqdm(
            self.valid_dataloader,
            desc=f"Validating Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            valid_loss_dict, valid_stats, total_valid_loss = self._valid_step(batch)
            epoch_sum_loss += total_valid_loss.item()

        self.accelerator.wait_for_everyone()
        return epoch_sum_loss / len(self.valid_dataloader)

    # TODO: train step
    def _train_step(self, data):
        global_step = self.step
        optimizer_idx = global_step % 2

        train_losses = {}
        total_loss = 0
        train_states = {}

        inputs = data["mel"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        reconstructions, posterior = self.model(inputs)
        # print(inputs.shape)
        # print(reconstructions.shape)
        # print(posterior.shape)
        # print(inputs.device)
        # print(reconstructions.device)
        # print(posterior.device)

        train_losses = self.criterion(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posterior,
            optimizer_idx=optimizer_idx,
            global_step=global_step,
            last_layer=self.model.get_last_layer(),
            split="train",
        )

        if optimizer_idx == 0:
            total_loss = train_losses["loss"]
            self.optimizer["opt_ae"].zero_grad()
            total_loss.backward()
            self.optimizer["opt_ae"].step()
            lr = self.optimizer["opt_ae"].param_groups[0]["lr"]

        else:
            total_loss = train_losses["d_loss"]
            self.optimizer["opt_disc"].zero_grad()
            total_loss.backward()
            self.optimizer["opt_disc"].step()
            lr = self.optimizer["opt_disc"].param_groups[0]["lr"]

        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        return train_losses, train_states, total_loss.item(), lr

    # TODO: eval step
    @torch.no_grad()
    def _valid_step(self, data):
        valid_loss = {}
        total_valid_loss = 0
        valid_stats = {}

        inputs = data["mel"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        reconstructions, posterior = self.model(inputs)

        loss = F.l1_loss(inputs, reconstructions)
        valid_loss["loss"] = loss

        total_valid_loss += loss

        for item in valid_loss:
            valid_loss[item] = valid_loss[item].item()

        return valid_loss, valid_stats, total_valid_loss

    def __dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )
