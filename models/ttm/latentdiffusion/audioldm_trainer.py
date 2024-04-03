# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.base.new_trainer import BaseTrainer
from diffusers import DDPMScheduler
from models.ttm.latentdiffusion.audioldm_dataset import AudioLDMDataset, AudioLDMCollator
from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.ldm.audioldm import AudioLDM, UNetModel
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from transformers import T5EncoderModel
from diffusers import DDPMScheduler
import os
import shutil
import torchaudio


class AudioLDMTrainer(BaseTrainer):
    # TODO: remove cfg_path
    def __init__(self, args, cfg, cfg_path=None):
        BaseTrainer.__init__(self, args, cfg)
        self.cfg = cfg

        self.build_autoencoderkl()
        self.build_textencoder()
        self.nosie_scheduler = self.build_noise_scheduler()
        
        self.debug_autoencoderkl_vocoder()

        # self.save_config_file()
    
    def debug_autoencoderkl_vocoder(self):
        import json
        from models.ttm.latentdiffusion.audioldm_inference import AttrDict
        from models.tta.ldm.inference_utils.vocoder import Generator
        import numpy as np
        # DEBUG
        debug_vae_dir = f"{self.exp_dir}/debug_vae_sample"
        shutil.rmtree(debug_vae_dir, ignore_errors=True)
        os.makedirs(debug_vae_dir, exist_ok=True)
        debug_vocoder_dir = f"{self.exp_dir}/debug_vocoder_sample"
        shutil.rmtree(debug_vocoder_dir, ignore_errors=True)
        os.makedirs(debug_vocoder_dir, exist_ok=True)
        config_file = os.path.join(self.cfg.model.vocoder_config_path)
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        self.vocoder = Generator(h).to(self.accelerator.device)
        checkpoint_dict = torch.load(
            self.cfg.model.vocoder_path, map_location=self.accelerator.device
        )
        self.vocoder.load_state_dict(checkpoint_dict["generator"])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()
        for i in range(min(8, len(self.train_dataset))):
            single_features = self.train_dataset[i]
            melspec = single_features["mel"][:, :624].unsqueeze(0).unsqueeze(1)
            melspec = melspec.to(self.accelerator.device)
            y_vocoder = self.vocoder(melspec.squeeze(0))
            audio_vocoder = y_vocoder.squeeze()
            audio_vocoder = audio_vocoder.cpu()
            audio_vocoder.unsqueeze_(0)
            torchaudio.save(f"{debug_vocoder_dir}/{single_features['caption'][:100].replace('/', '-')}_vocoder.wav", audio_vocoder, self.cfg.preprocess.sample_rate)
            latent = self.mel_to_latent(melspec)
            with torch.no_grad():
                mel_out = self.autoencoderkl.decode(latent)
                melspec = mel_out[0, 0].cpu().detach().numpy()
                melspec = np.expand_dims(melspec, 0)
                melspec = torch.FloatTensor(melspec).to(self.accelerator.device)
                y = self.vocoder(melspec)
                audio = y.squeeze()
                # audio = audio * 32768.0
                audio = audio.cpu()
                audio.unsqueeze_(0)
                torchaudio.save(f"{debug_vae_dir}/{single_features['caption'][:100].replace('/', '-')}_recon.wav", audio, self.cfg.preprocess.sample_rate)

    def build_autoencoderkl(self):
        self.autoencoderkl = AutoencoderKL(self.cfg.model.autoencoderkl)
        self.autoencoder_path = self.cfg.model.autoencoder_path
        # load model
        ckpt = torch.load(os.path.join(self.autoencoder_path))
        self.autoencoderkl.load_state_dict(ckpt)
        self.autoencoderkl.to(self.accelerator.device)
        self.autoencoderkl.requires_grad_(requires_grad=False)
        self.autoencoderkl.eval()

    def build_textencoder(self):
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.text_encoder.to(self.accelerator.device)
        self.text_encoder.requires_grad_(requires_grad=False)
        self.text_encoder.eval()

    def build_noise_scheduler(self):
        nosie_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg.model.noise_scheduler.num_train_timesteps,
            beta_start=self.cfg.model.noise_scheduler.beta_start,
            beta_end=self.cfg.model.noise_scheduler.beta_end,
            beta_schedule=self.cfg.model.noise_scheduler.beta_schedule,
            clip_sample=self.cfg.model.noise_scheduler.clip_sample,
            # steps_offset=self.cfg.model.noise_scheduler.steps_offset,
            # set_alpha_to_one=self.cfg.model.noise_scheduler.set_alpha_to_one,
            # skip_prk_steps=self.cfg.model.noise_scheduler.skip_prk_steps,
            prediction_type=self.cfg.model.noise_scheduler.prediction_type,
        )
        return nosie_scheduler

    def _build_dataset(self):
        return AudioLDMDataset, AudioLDMCollator

    def _build_dataloader(self):
        Dataset, Collator = self._build_dataset()
        # build dataset instance for each dataset and combine them by ConcatDataset
        datasets_list = []
        for dataset in self.cfg.dataset:
            subdataset = Dataset(self.cfg, dataset, is_valid=False)
            datasets_list.append(subdataset)
        train_dataset = ConcatDataset(datasets_list)
        self.train_dataset = train_dataset

        # DEBUG
        debug_sample_dir = f"{self.exp_dir}/debug_train_sample"
        shutil.rmtree(debug_sample_dir, ignore_errors=True)
        os.makedirs(debug_sample_dir, exist_ok=True)
        with open(f"{debug_sample_dir}/info", "w") as f:
            for i in range(min(8, len(train_dataset))):
                single_features = train_dataset[i]
                # print(wavs.shape)
                # print(infos)
                f.write(f"{single_features['caption']}\n")
                # print(infos.to_condition_attributes())
                torchaudio.save(f"{debug_sample_dir}/{single_features['caption'][:100].replace('/', '-')}.wav", single_features['wav'], self.cfg.preprocess.sample_rate)

        train_collate = Collator(self.cfg)

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
        data_loader = {"train": train_loader, "valid": valid_loader}
        return train_loader, valid_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.train.adam)
        return optimizer

    # TODO: check it...
    def _build_scheduler(self):
        return None
        # return ReduceLROnPlateau(self.optimizer["opt_ae"], **self.cfg.train.lronPlateau)

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def _build_criterion(self):
        criterion = nn.MSELoss(reduction="mean")
        return criterion

    def get_state_dict(self):
        if self.scheduler != None:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "batch_size": self.cfg.train.batch_size,
            }
        else:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "batch_size": self.cfg.train.batch_size,
            }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler != None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _build_model(self):
        self.model = AudioLDM(self.cfg.model.audioldm)
        return self.model

    @torch.no_grad()
    def mel_to_latent(self, melspec):
        posterior = self.autoencoderkl.encode(melspec)
        latent = posterior.sample()  # (B, 4, 5, 78)
        return latent

    @torch.no_grad()
    def get_text_embedding(self, text_input_ids, text_attention_mask):
        text_embedding = self.text_encoder(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state
        return text_embedding  # (B, T, 768)

    def _train_step(self, data):

        melspec = data["mel"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        latents = self.mel_to_latent(melspec)

        text_embedding = self.get_text_embedding(
            data["text_input_ids"], data["text_attention_mask"]
        )

        noise = torch.randn_like(latents).float()

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.cfg.model.noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        with torch.no_grad():
            noisy_latents = self.nosie_scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.model(
            noisy_latents, timesteps=timesteps, context=text_embedding
        )

        loss = self.criterion(model_pred, noise)

        return loss

    # TODO: eval step
    @torch.no_grad()
    def _valid_step(self, data):

        melspec = data["mel"].unsqueeze(1)  # (B, 80, T) -> (B, 1, 80, T)
        latents = self.mel_to_latent(melspec)

        text_embedding = self.get_text_embedding(
            data["text_input_ids"], data["text_attention_mask"]
        )

        noise = torch.randn_like(latents).float()

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.cfg.model.noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        noisy_latents = self.nosie_scheduler.add_noise(latents, noise, timesteps)

        model_pred = self.model(noisy_latents, timesteps, text_embedding)

        loss = self.criterion(model_pred, noise)

        return loss
