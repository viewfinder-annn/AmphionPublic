# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import json
from utils.data_utils import *


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
    def __init__(self, text, wav, device, sample_rate):
        self.text:dict = text
        self.wav:dict = wav
        self.device = device
        self.sample_rate = sample_rate
    # Plugin to the MusicGen Conditioning Mechanism
    def to_condition_attributes(self) -> ConditioningAttributes:
        res = ConditioningAttributes()
        res.text = self.text
        for k, v in self.wav.items():
            res.wav[k] = WavCondition(wav=v, length=torch.tensor([v.shape[-1]], device=self.device), sample_rate=[self.sample_rate])
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
            self.metadata = self.metadata[:min(num_samples, len(self.metadata))]
        
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
        
        # self_wav
        self_wav, sr = audio_read(utt_info["self_wav"])
        self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
        wav_condition["self_wav"] = self_wav

        # caption
        if self.cfg.preprocess.use_caption:
            caption = utt_info["caption"]
            text_condition["description"] = caption

        # ref_wav
        if self.cfg.preprocess.use_ref_wav:
            ref_wav, sr = audio_read(utt_info["ref_wav"])
            ref_wav = convert_audio(ref_wav, sr, self.sample_rate, self.preprocess.audio_channels)
            wav_condition["ref_wav"] = ref_wav

        condition_info = ConditionInfo(text_condition, wav_condition, self.device, self.sample_rate)
        
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