# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import json
from utils.data_utils import *
import librosa


from models.base.base_dataset import (
    BaseOnlineDataset,
    BaseOnlineCollator,
    BaseTestDataset,
    BaseTestCollator,
)
import librosa

from modules.audiocraft.modules.conditioners import (
    ConditioningAttributes,
    JointEmbedCondition,
    WavCondition,
)

from modules.audiocraft.data.audio import audio_read
from modules.audiocraft.data.audio_utils import convert_audio

class ConditionInfo:
    def __init__(self, text, wav, wav_path, device, sample_rate):
        self.text:dict = text
        self.wav:dict = wav
        self.wav_path:dict = wav_path
        self.device = device
        self.sample_rate = sample_rate
    # Plugin to the MusicGen Conditioning Mechanism
    def to_condition_attributes(self) -> ConditioningAttributes:
        res = ConditioningAttributes()
        res.text = self.text
        for k, v in self.wav.items():
            # unsqueeze: required in audiocraft/modules/conditioners.py, see _collate_wavs
            res.wav[k] = WavCondition(wav=v.unsqueeze(0), length=torch.tensor([v.shape[-1]], device=self.device), sample_rate=[self.sample_rate])
        return res

class MusicGenDataset:
    def __init__(self, cfg, dataset, is_valid=False):
        # BaseOnlineDataset.__init__(self, cfg, dataset, is_valid=is_valid)

        self.cfg = cfg
        self.device = self.cfg.audiocraft.device
        self.sample_rate = self.cfg.preprocess.sample_rate
        
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()
        num_samples = cfg.train.valid_num_samples if is_valid else cfg.train.train_num_samples
        if num_samples != -1:
            self.metadata = random.sample(self.metadata, min(num_samples, len(self.metadata)))
        
        """
        metadata: a list contains dict, each dict contains:
            dataset(str): dataset name
            caption(str): caption
            ref_wav(str): reference wav path(for example vocal)
            self_wav(str): self wav path(for example accompaniment, music want to generate)
        """

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        text_condition = {}
        wav_condition = {}
        wav_path = {}
        
        # self_wav, comply with msd data
        # self_wav, sr = librosa.load(utt_info["self_wav"], sr=None, mono=False)
        # self_wav = torch.tensor(self_wav, dtype=torch.float32, device=self.device)
        # if self_wav.dim() == 1:
        #     self_wav = self_wav.unsqueeze(0)
        if "start" in utt_info:
            self_wav, sr = audio_read(utt_info["self_wav"], seek_time=utt_info["start"], duration=self.cfg.preprocess.segment_duration, pad=True)
        else:
            self_wav, sr = audio_read(utt_info["self_wav"], duration=self.cfg.preprocess.segment_duration, pad=True)
        if self_wav.dim() == 1:
            self_wav = self_wav.unsqueeze(0)
        self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
        # for wavs pad to self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate
        # if self_wav.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
        #     self_wav = torch.cat([self_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - self_wav.shape[-1]], dtype=self_wav.dtype, device=self_wav.device)], dim=-1)
        # else:
        #     self_wav = self_wav[:, :self.cfg.preprocess.segment_duration * self.sample_rate]
        wav_condition["self_wav"] = self_wav
        wav_path["self_wav"] = utt_info["self_wav"]

        # caption
        if self.cfg.preprocess.use_caption:
            caption = utt_info["caption"]
            text_condition["description"] = caption
        
        if self.cfg.preprocess.use_lyric:
            lyric = utt_info["lyric"]
            text_condition["lyric"] = lyric

        # ref_wav
        if self.cfg.preprocess.use_ref_wav:
            ref_wav, sr = audio_read(utt_info["ref_wav"])
            ref_wav = convert_audio(ref_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
            if ref_wav.shape[-1] <= self.cfg.preprocess.segment_duration * self.sample_rate:
                ref_wav = torch.cat([ref_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - ref_wav.shape[-1]], dtype=ref_wav.dtype, device=ref_wav.device)], dim=-1)
            else:
                ref_wav = ref_wav[:, :self.cfg.preprocess.segment_duration * self.sample_rate]
            wav_condition["ref_wav"] = ref_wav
            wav_path["ref_wav"] = utt_info["ref_wav"]

        condition_info = ConditionInfo(text_condition, wav_condition, wav_path, self.device, self.sample_rate)
        
        return self_wav, condition_info

    def __len__(self):
        return len(self.metadata)

class MusicGenCollator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        wavs = torch.stack([x[0] for x in batch])
        condition_infos = [x[1] for x in batch]
        return wavs, condition_infos