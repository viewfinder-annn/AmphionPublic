# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
import json
from pathlib import Path
import re

from models.tta.autoencoder.autoencoder import AutoencoderKL
from models.tta.ldm.inference_utils.vocoder import Generator
from models.tta.ldm.audioldm import AudioLDM
from transformers import T5EncoderModel, AutoTokenizer
from diffusers import PNDMScheduler

import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import accelerate
from utils.util import Logger

from models.ttm.latentdiffusion.mel import mel_spectrogram
import json5
import torchaudio
from modules.audiocraft.data.audio_utils import convert_audio

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AudioLDMInference:
    def __init__(self, args, cfg, cfg_path=None):
        self.cfg = cfg
        self.args = args
        
        self.out_path = self.args.output_dir
        log_file = os.path.join(self.args.infer_expt_dir, "infer.log")
        self.logger = Logger(log_file).logger

        self.model = self.build_model()
        
        # init with accelerate
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self.accelerator = accelerate.Accelerator()
        self.model = self.accelerator.prepare(self.model)
        end = time.monotonic_ns()
        self.accelerator.wait_for_everyone()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.3f}ms")
    
        self.build_autoencoderkl()
        self.build_textencoder()
        self.build_vocoder()
        
        with self.accelerator.main_process_first():
            self.logger.info("Loading checkpoint...")
            start = time.monotonic_ns()
            # TODO: Also, suppose only use latest one yet
            self.checkpoint_path = self.__load_model(os.path.join(self.args.infer_expt_dir, "checkpoint"))
            end = time.monotonic_ns()
            self.logger.info(f"Loading checkpoint done in {(end - start) / 1e6:.3f}ms")

        self.model.eval()
        self.accelerator.wait_for_everyone()
        
        # self.generation_params = {
        #     'use_sampling': False,
        #     'temp': 0,
        #     'top_k': 0,
        #     'top_p': 0,
        #     'cfg_coef': 3.0,
        #     'two_step_cfg': self.cfg_omega.transformer_lm.two_step_cfg,
        # }

        checkpoint_name = os.path.basename(self.checkpoint_path)
        self.out_path = os.path.join(self.args.output_dir, checkpoint_name)
        self.out_mel_path = os.path.join(self.out_path, "mel")
        self.out_wav_path = os.path.join(self.out_path, "wav")
        os.makedirs(self.out_mel_path, exist_ok=True)
        os.makedirs(self.out_wav_path, exist_ok=True)

    def build_autoencoderkl(self):
        self.autoencoderkl = AutoencoderKL(self.cfg.model.autoencoderkl)
        self.autoencoder_path = self.cfg.model.autoencoder_path
        # load model
        ckpt = torch.load(os.path.join(self.autoencoder_path))
        self.autoencoderkl.load_state_dict(ckpt)
        # self.autoencoderkl.load_state_dict(ckpt["model"])
        self.autoencoderkl.to(self.accelerator.device)
        self.autoencoderkl.requires_grad_(requires_grad=False)
        self.autoencoderkl.eval()

    def build_textencoder(self):
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.text_encoder.cuda(self.accelerator.device)
        self.text_encoder.requires_grad_(requires_grad=False)
        self.text_encoder.eval()

    def build_vocoder(self):
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

    def build_model(self):
        self.model = AudioLDM(self.cfg.model.audioldm)
        return self.model

    def __load_model(self, checkpoint_dir: str = None, checkpoint_path: str = None):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            ls = []
            for i in Path(checkpoint_dir).iterdir():
                if re.match(r"epoch-\d+_step-\d+_loss-[\d.]+", str(i.stem)):
                    ls.append(i)
            ls.sort(
                key=lambda x: int(x.stem.split("_")[-2].split("-")[-1]), reverse=True
            )
            checkpoint_path = ls[0]
        else:
            checkpoint_path = Path(checkpoint_path)
        self.accelerator.load_state(str(checkpoint_path))
        self.logger.info(f"Load checkpoint from {checkpoint_path}")
        # set epoch and step
        self.epoch = int(checkpoint_path.stem.split("_")[-3].split("-")[-1])
        self.step = int(checkpoint_path.stem.split("_")[-2].split("-")[-1])
        return str(checkpoint_path)

    def get_text_embedding(self, text):

        prompt = text

        text_input = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            padding="do_not_pad",
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.accelerator.device)
        )[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * 1, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.accelerator.device)
        )[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def generate(self, descriptions, waveforms):
        # TODO: waveforms
        text_embeddings = self.get_text_embedding(descriptions)
        print(text_embeddings.shape)

        num_steps = self.cfg.inference.num_steps
        guidance_scale = self.cfg.inference.guidance_scale

        noise_scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )

        noise_scheduler.set_timesteps(num_steps)

        latents = torch.randn(
            (
                1,
                self.cfg.model.autoencoderkl.z_channels,
                80 // (2 ** (len(self.cfg.model.autoencoderkl.ch_mult) - 1)),
                624 // (2 ** (len(self.cfg.model.autoencoderkl.ch_mult) - 1)),
            )
        ).to(self.accelerator.device)

        self.model.eval()
        for t in tqdm(noise_scheduler.timesteps):
            t = t.to(self.accelerator.device)

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )
            # print(latent_model_input.shape)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.model(
                    latent_model_input, torch.cat([t.unsqueeze(0)] * 2), text_embeddings
                )

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            # print(latents.shape)

        latents_out = latents
        print(latents_out.shape)

        with torch.no_grad():
            mel_out = self.autoencoderkl.decode(latents_out)
        print(mel_out.shape)

        melspec = mel_out[0, 0].cpu().detach().numpy()
        plt.imsave(os.path.join(self.out_mel_path, descriptions[0][:100] + ".png"), melspec)

        self.vocoder.eval()
        with torch.no_grad():
            melspec = np.expand_dims(melspec, 0)
            melspec = torch.FloatTensor(melspec).to(self.accelerator.device)

            y = self.vocoder(melspec)
            audio = y.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype("int16")

        write(os.path.join(self.out_wav_path, descriptions[0][:100] + ".wav"), 16000, audio)
        return audio
    
    def waveform_to_mel(self, waveforms):
        melspec = mel_spectrogram(
            waveforms,
            n_fft=self.cfg.preprocess.n_fft,
            num_mels=self.cfg.preprocess.n_mel,
            sampling_rate=self.cfg.preprocess.sample_rate,
            hop_size=self.cfg.preprocess.hop_size,
            win_size=self.cfg.preprocess.win_size,
            fmin=self.cfg.preprocess.fmin,
            fmax=self.cfg.preprocess.fmax,
        )
        return melspec
    
    @torch.no_grad()
    def mel_to_latent(self, melspec):
        posterior = self.autoencoderkl.encode(melspec)
        latent = posterior.sample()  # (B, 4, 5, 78)
        return latent
    
    def generate_from_ref(self, waveform):
        # 使用VAE编码器将waveforms转换为latents
        ref_melspec = self.waveform_to_mel(waveform)[:, :, :624].unsqueeze(1)
        ref_melspec = ref_melspec.to(self.accelerator.device)
        print(ref_melspec.shape)
        ref_latents = self.mel_to_latent(ref_melspec)

        print(ref_latents.shape)

        num_steps = self.cfg.inference.num_steps

        noise_scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )

        noise_scheduler.set_timesteps(num_steps)

        # 生成初始噪声
        noise = torch.randn_like(ref_latents).float()

        self.model.eval()
        # for t in tqdm(noise_scheduler.timesteps):
        for t in noise_scheduler.timesteps:
            latents = torch.cat([noise, ref_latents], dim=1)
            
            t = t.to(self.accelerator.device)

            # 缩放模型输入
            latent_model_input = noise_scheduler.scale_model_input(latents, timestep=t)

            # 预测噪声残差
            with torch.no_grad():
                noise_pred = self.model(latent_model_input, timesteps=t.unsqueeze(0))

            # 计算先前的噪声样本 x_t -> x_t-1
            noise = noise_scheduler.step(noise_pred, t, noise).prev_sample

        latents_out = noise
        print(latents_out.shape)

        # 解码生成的latents
        with torch.no_grad():
            mel_out = self.autoencoderkl.decode(latents_out)
        print(mel_out.shape)

        melspec = mel_out[0, 0].cpu().detach().numpy()
        # plt.imsave(os.path.join(self.out_mel_path, "generated.png"), melspec)

        # 使用vocoder生成音频
        self.vocoder.eval()
        with torch.no_grad():
            melspec = np.expand_dims(melspec, 0)
            melspec = torch.FloatTensor(melspec).to(self.accelerator.device)

            y = self.vocoder(melspec)
            audio = y.squeeze()
            # audio_vocoder = y_vocoder.squeeze()
            # audio_vocoder = audio_vocoder.cpu()
            audio = audio.cpu()
            audio.unsqueeze_(0)

        # write(os.path.join(self.out_wav_path, "generated.wav"), 16000, audio)
        return audio, melspec
    
    def inference(self):
        texts = json5.load(open(self.args.text))
        for meta in tqdm(texts):
            self_wav, self_sr = torchaudio.load(meta["self_wav"])
            ref_wav, ref_sr = torchaudio.load(meta["ref_wav"])
            self_shape, ref_shape = self_wav.shape[-1], ref_wav.shape[-1]
            ref_wav = convert_audio(ref_wav, ref_sr, self.cfg.preprocess.sample_rate, self.cfg.preprocess.audio_channels)
            if ref_wav.shape[-1] < self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate:
                print(f"padding {meta['ref_wav']} from {ref_wav.shape[-1]/self_sr} to {self.cfg.preprocess.segment_duration} seconds")
                ref_wav = torch.cat([ref_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate - ref_wav.shape[-1]], dtype=ref_wav.dtype, device=ref_wav.device)], dim=-1)
            elif ref_wav.shape[-1] > self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate:
                print(f"cutting {meta['ref_wav']} from {ref_wav.shape[-1]/self_sr} to {self.cfg.preprocess.segment_duration} seconds")
                ref_wav = ref_wav[:, :self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate]
            self_wav = convert_audio(self_wav, self_sr, self.cfg.preprocess.sample_rate, self.cfg.preprocess.audio_channels)
            if self_wav.shape[-1] < self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate:
                print(f"padding {meta['self_wav']} from {self_wav.shape[-1]/self_sr} to {self.cfg.preprocess.segment_duration} seconds")
                self_wav = torch.cat([self_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate - self_wav.shape[-1]], dtype=self_wav.dtype, device=self_wav.device)], dim=-1)
            elif self_wav.shape[-1] > self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate:
                print(f"cutting {meta['self_wav']} from {self_wav.shape[-1]/self_sr} to {self.cfg.preprocess.segment_duration} seconds")
                self_wav = self_wav[:, :self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate]
            
            audio, melspec = self.generate_from_ref(ref_wav)
            if audio.shape[-1] < self_wav.shape[-1]:
                audio = torch.cat([audio, torch.zeros([self.cfg.preprocess.audio_channels, self_wav.shape[-1] - audio.shape[-1]], dtype=audio.dtype, device=audio.device)], dim=-1)
            print(audio.shape, ref_wav.shape, self_wav.shape)
            audio_mix = ref_wav + audio
            
            if ref_shape < self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate:
                audio = audio[:, :ref_shape]
                audio_mix = audio_mix[:, :ref_shape]
                ref_wav = ref_wav[:, :ref_shape]
                self_wav = self_wav[:, :ref_shape]
            
            torchaudio.save(os.path.join(self.out_wav_path, meta["self_wav"].split("/")[-1].replace(".wav", "_generated.wav")), audio, self.cfg.preprocess.sample_rate)
            torchaudio.save(os.path.join(self.out_wav_path, meta["self_wav"].split("/")[-1].replace(".wav", "_mix.wav")), audio_mix, self.cfg.preprocess.sample_rate)
            torchaudio.save(os.path.join(self.out_wav_path, meta["self_wav"].split("/")[-1].replace(".wav", "_gt.wav")), self_wav+ref_wav, self.cfg.preprocess.sample_rate)
            torchaudio.save(os.path.join(self.out_wav_path, meta["ref_wav"].split("/")[-1].replace(".wav", "_input.wav")), ref_wav, self.cfg.preprocess.sample_rate)
            
        
    def inference_old(self):    
        audios = []
        texts = None
        waveforms = None
        if self.args.file_list:
            if self.args.text != "":
                with open(self.args.text, "r") as f:
                    texts = f.readlines()
                    texts = [t.strip() for t in texts]
            if self.args.waveform != "":
                with open(self.args.waveform, "r") as f:
                    waveforms = f.readlines()
                    waveforms = [w.strip() for w in waveforms]
        else:
            if self.args.text != "":
                texts = [self.args.text]
            if self.args.waveform != "":
                waveforms = [self.args.waveform]
        
        assert texts is not None or waveforms is not None, "Either text or waveform should be provided"
        
        if waveforms is not None and texts is not None:
            assert len(waveforms) == len(texts), "Number of waveforms and texts doesn't match"
            if len(texts) > 1:
                for i in tqdm(range(0, len(texts), self.cfg.inference.batch_size), desc="Generating"):
                    batch_text = texts[i:i+self.cfg.inference.batch_size]
                    batch_waveform = waveforms[i:i+self.cfg.inference.batch_size]
                    audios.extend(self.generate(batch_text, batch_waveform))
            else:
                audios = self.generate(texts, waveforms)
        elif waveforms is not None and texts is None:
            if len(waveforms) > 1:
                for i in tqdm(range(0, len(waveforms), self.cfg.inference.batch_size), desc="Generating"):
                    batch_waveform = waveforms[i:i+self.cfg.inference.batch_size]
                    audios.extend(self.generate(texts, batch_waveform))
            else:
                audios = self.generate(texts, waveforms)
        elif waveforms is None and texts is not None:
            if len(texts) > 1:
                for i in tqdm(range(0, len(texts), self.cfg.inference.batch_size), desc="Generating"):
                    batch_text = texts[i:i+self.cfg.inference.batch_size]
                    audios.extend(self.generate(batch_text, waveforms))
            else:
                audios = self.generate(texts, waveforms)
