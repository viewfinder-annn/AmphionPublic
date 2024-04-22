# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/genmodel.py

import modules.audiocraft.solvers.builders as builders
import modules.audiocraft.models as models
from modules.audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
from modules.audiocraft.solvers.compression import CompressionSolver
from modules.audiocraft.utils.autocast import TorchAutocast
from modules.audiocraft.data.audio_utils import convert_audio
import accelerate
import omegaconf
import json5
import torch
import re
import os
import time
import shutil
from tqdm import tqdm
from pathlib import Path
import torchaudio
from utils.util import Logger

class Sing2SongInference:
    DATASET_TYPE = builders.DatasetType.MUSIC
    
    def __init__(self, args=None, cfg=None, cfg_path=None):
        self.args = args
        self.cfg = cfg
        
        self.out_path = self.args.output_dir
        log_file = os.path.join(self.args.infer_expt_dir, "infer.log")
        self.logger = Logger(log_file).logger
        
        # print(type(cfg))
        # exit()
        # for audiocraft compatibility
        self.cfg_omega = omegaconf.OmegaConf.create(json5.load(open(cfg_path))['audiocraft'])
        self.device = self.cfg_omega.device
        self.autocast_dtype = {
            'float16': torch.float16, 'bfloat16': torch.bfloat16
        }[self.cfg_omega.autocast_dtype]
        self.autocast = TorchAutocast(enabled=self.cfg_omega.autocast, device_type=self.device, dtype=self.autocast_dtype)
        
        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg_omega, self.cfg_omega.compression_model_checkpoint, device=self.device)
        self.compression_model.eval()
        
        self.model = self._build_model()
        # TODO: remove omega
        max_duration = self.cfg_omega.dataset.segment_duration  # type: ignore
        assert max_duration is not None
        self.max_duration: float = max_duration
        self.duration = self.max_duration
        
        # init with accelerate
        self.logger.info("Initializing accelerate...")
        start = time.monotonic_ns()
        self.accelerator = accelerate.Accelerator()
        self.model = self.accelerator.prepare(self.model)
        end = time.monotonic_ns()
        self.accelerator.wait_for_everyone()
        self.logger.info(f"Initializing accelerate done in {(end - start) / 1e6:.3f}ms")
        
        with self.accelerator.main_process_first():
            self.logger.info("Loading checkpoint...")
            start = time.monotonic_ns()
            # TODO: Also, suppose only use latest one yet
            self.checkpoint_path = self.__load_model(os.path.join(self.args.infer_expt_dir, "checkpoint"))
            end = time.monotonic_ns()
            self.logger.info(f"Loading checkpoint done in {(end - start) / 1e6:.3f}ms")

        self.model.eval()
        self.accelerator.wait_for_everyone()
        
        self.generation_params = self.cfg.inference.generation_params
    
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
        if self.cfg_omega.fsdp.use:
            assert not self.cfg_omega.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model)
        
        # ema
        # self.register_ema('model')
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
                key=lambda x: int(x.stem.split("_")[-3].split("-")[-1]), reverse=True
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
    
    
    @property
    def frame_rate(self) -> float:
        """Roughly the number of AR steps per seconds."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of the generated audio."""
        return self.compression_model.channels

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            action,
            category,
            prompt,
    ):
        """Prepare model inputs.

        Args:
            action (list of str): A list of strings used as action conditioning.
            category (list of str): A list of strings used as category conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
        """
        attributes = [
            ConditioningAttributes(text={'action': action, 'category': category})
        ]

        if prompt is not None:
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def generate(self, meta, progress: bool = False, return_tokens: bool = False):
        """Generate samples conditioned on text.

        Args:
            meta: a dict contains: action(str), category(str), ref_wav(str), self_wav(str)
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        action, category = meta['action'], meta['category']
        waveform_tensor, waveform_sr = torchaudio.load(meta['ref_wav'])
        waveform_tensor = convert_audio(waveform_tensor, waveform_sr, self.sample_rate, self.audio_channels)
        if waveform_tensor.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
            waveform_tensor = torch.cat([waveform_tensor, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - waveform_tensor.shape[-1]], dtype=waveform_tensor.dtype, device=waveform_tensor.device)], dim=-1)
        # add batch dimension
        waveform_tensor = waveform_tensor.unsqueeze(0)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(action, category, waveform_tensor)
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        # remove prompt tokens
        tokens = tokens[:, :, prompt_tokens.shape[-1]:]
        return self.generate_audio(tokens), self.generate_audio(prompt_tokens)
    
    def _generate_tokens(self, attributes, prompt_tokens, progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (here text).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        if prompt_tokens is not None:
            total_gen_len = prompt_tokens.shape[-1] + int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.model.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            stride_tokens = int(self.frame_rate * self.extend_stride)
            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                with self.autocast:
                    gen_tokens = self.model.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

    def generate_audio(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        """Generate Audio from tokens."""
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio
    
    def inference(self):
        """
        args.text: file to be synthesized, which contains a list of meta(ref_wav, action, category)
        """
        texts = json5.load(open(self.args.text))
        checkpoint_name = os.path.basename(self.checkpoint_path)
        target_dir = os.path.join(self.args.output_dir, f"{checkpoint_name}_{self.generation_params.use_sampling}_{self.generation_params.temp}_{self.generation_params.top_k}_{self.generation_params.top_p}")
        os.makedirs(target_dir, exist_ok=True)
        gt_dir = os.path.join(target_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        origin_dir = os.path.join(target_dir, "origin")
        os.makedirs(origin_dir, exist_ok=True)
        gt_mix_dir = os.path.join(target_dir, "gt_mix")
        os.makedirs(gt_mix_dir, exist_ok=True)
        mix_dir = os.path.join(target_dir, "mix")
        os.makedirs(mix_dir, exist_ok=True)
        output_dir = os.path.join(target_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # TODO: batched inference
        for meta in tqdm(texts, desc="Generating"):
            audio, prompt_audio = self.generate(meta)
            # remove batch dimension
            audio.squeeze_(0)
            prompt_audio.squeeze_(0)
            
            ref_file = os.path.join(origin_dir, f"{os.path.splitext(os.path.basename(meta['ref_wav']))[0]}.wav")
            torchaudio.save(ref_file, prompt_audio.cpu(), self.sample_rate)
            
            file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(meta['ref_wav']))[0] + '+' + meta['action'] + '_' + meta['category']}.wav")
            torchaudio.save(file, audio.cpu(), self.sample_rate)
            
            gt_file = os.path.join(gt_dir, f"{os.path.splitext(os.path.basename(meta['ref_wav']))[0] + '+' + meta['action'] + '_' + meta['category']}_gt.wav")
            gt_audio, gt_sr = torchaudio.load(meta['self_wav'])
            gt_audio_converted = convert_audio(gt_audio, gt_sr, self.sample_rate, self.audio_channels)
            torchaudio.save(gt_file, gt_audio_converted, self.sample_rate)
            
            if meta["action"] == "add":
                mix = prompt_audio.cpu() + audio.cpu()
                gt_mix = prompt_audio.cpu() + gt_audio_converted
            elif meta["action"] == "extract":
                mix = prompt_audio.cpu() - audio.cpu()
                gt_mix = prompt_audio.cpu() - gt_audio_converted
            elif meta["action"] == "remove":
                mix = audio.cpu()
                gt_mix = gt_audio_converted
            file = os.path.join(mix_dir, f"{os.path.splitext(os.path.basename(meta['ref_wav']))[0] + '+' + meta['action'] + '_' + meta['category']}_mix.wav")
            torchaudio.save(file, mix.cpu(), self.sample_rate)
            gt_file = os.path.join(gt_mix_dir, f"{os.path.splitext(os.path.basename(meta['ref_wav']))[0] + '+' + meta['action'] + '_' + meta['category']}_gt_mix.wav")
            torchaudio.save(gt_file, gt_mix.cpu(), self.sample_rate)