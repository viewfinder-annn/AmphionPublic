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
from tqdm import tqdm
from pathlib import Path
import torchaudio
from utils.util import Logger

class MusicGenInference:
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
        
        self.generation_params = {
            'use_sampling': False,
            'temp': 0,
            'top_k': 0,
            'top_p': 0,
            'cfg_coef': 3.0,
            'two_step_cfg': self.cfg_omega.transformer_lm.two_step_cfg,
        }
    
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
            descriptions,
            waveforms,
            prompt,
    ):
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            waveforms (list of torch.Tensor): A batch of waveforms used for waveform condition.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
        """
        print(descriptions)
        print(waveforms)
        if descriptions is not None:
            attributes = [
                ConditioningAttributes(text={'description': description, 'lyric': description})
                for description in descriptions]
            if waveforms is not None:
                waveforms_tensor = [torchaudio.load(waveform_path) for waveform_path in waveforms]
                waveforms_normalized = []
                for waveform in waveforms_tensor:
                    waveform_normalized = convert_audio(waveform[0], waveform[1], self.sample_rate, self.audio_channels)
                    if waveform_normalized.shape[-1] < self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate:
                        waveform_normalized = torch.cat([waveform_normalized, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - waveform_normalized.shape[-1]], dtype=waveform_normalized.dtype, device=waveform_normalized.device)], dim=-1)
                    waveforms_normalized.append(waveform_normalized)
                for i, waveform in enumerate(waveforms_normalized):
                    attributes[i].wav = {'ref_wav': WavCondition(wav=waveform.unsqueeze(0).to(self.device), length=torch.tensor([waveform.shape[-1]], device=self.device), sample_rate=[self.sample_rate])}
        else:
            waveforms_tensor = [torchaudio.load(waveform_path) for waveform_path in waveforms]
            waveforms_normalized = []
            for waveform in waveforms_tensor:
                waveform_normalized = convert_audio(waveform[0], waveform[1], self.sample_rate, self.audio_channels)
                if waveform_normalized.shape[-1] < self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate:
                    waveform_normalized = torch.cat([waveform_normalized, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - waveform_normalized.shape[-1]], dtype=waveform_normalized.dtype, device=waveform_normalized.device)], dim=-1)
                waveforms_normalized.append(waveform_normalized)
            # C, T -> [1, C, T]
            attributes = [
                ConditioningAttributes(wav={'ref_wav': WavCondition(wav=waveform.unsqueeze(0).to(self.device), length=torch.tensor([waveform.shape[-1]], device=self.device), sample_rate=[self.sample_rate])})
                for waveform in waveforms_normalized]

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def generate(self, descriptions, waveforms, progress: bool = False, return_tokens: bool = False):
        """Generate samples conditioned on text.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, waveforms, None)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
    
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
            print(111)
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.model.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            print(222)
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
        
        checkpoint_name = os.path.basename(self.checkpoint_path)
        target_dir = os.path.join(self.args.output_dir, checkpoint_name)
        os.makedirs(target_dir, exist_ok=True)
        
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
            for i, audio in enumerate(audios):
                file = os.path.join(target_dir, f"{os.path.basename(waveforms[i]) + '=-=' + texts[i][:50]}.wav")
                torchaudio.save(file, audio.cpu(), self.sample_rate)
                # TODO: if self.args.mix:
                waveform_tensor, waveform_sr = torchaudio.load(waveforms[i])
                waveform_tensor = convert_audio(waveform_tensor, waveform_sr, self.sample_rate, self.audio_channels)
                mix = waveform_tensor + audio.cpu()
                file = os.path.join(target_dir, f"{os.path.basename(waveforms[i]) + '=-=' + texts[i][:50]}_mix.wav")
                torchaudio.save(file, mix.cpu(), self.sample_rate)
        elif waveforms is not None and texts is None:
            if len(waveforms) > 1:
                for i in tqdm(range(0, len(waveforms), self.cfg.inference.batch_size), desc="Generating"):
                    batch_waveform = waveforms[i:i+self.cfg.inference.batch_size]
                    audios.extend(self.generate(texts, batch_waveform))
            else:
                audios = self.generate(texts, waveforms)
            for i, audio in enumerate(audios):
                file = os.path.join(target_dir, f"{os.path.basename(waveforms[i])}.wav")
                torchaudio.save(file, audio.cpu(), self.sample_rate)
                # TODO: if self.args.mix:
                waveform_tensor, waveform_sr = torchaudio.load(waveforms[i])
                waveform_tensor = convert_audio(waveform_tensor, waveform_sr, self.sample_rate, self.audio_channels)
                mix = waveform_tensor + audio.cpu()
                file = os.path.join(target_dir, f"{os.path.basename(waveforms[i])}_mix.wav")
                torchaudio.save(file, mix.cpu(), self.sample_rate)
        elif waveforms is None and texts is not None:
            if len(texts) > 1:
                for i in tqdm(range(0, len(texts), self.cfg.inference.batch_size), desc="Generating"):
                    batch_text = texts[i:i+self.cfg.inference.batch_size]
                    # audios.extend(self.generate(batch_text, waveforms))
                    for idx, audio in enumerate(self.generate(batch_text, waveforms)):
                        file = os.path.join(target_dir, f"{i+idx}.wav")
                        torchaudio.save(file, audio.cpu(), self.sample_rate)
                        with open(file.replace(".wav", ".txt"), "w") as f:
                            f.write(batch_text[idx])
            else:
                audios = self.generate(texts, waveforms)
                for i, audio in enumerate(self.generate(texts, waveforms)):
                    file = os.path.join(target_dir, f"{i}.wav")
                    torchaudio.save(file, audio.cpu(), self.sample_rate)
                    with open(file.replace(".wav", ".txt"), "w") as f:
                        f.write(batch_text[i])