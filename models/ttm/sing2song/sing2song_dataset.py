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
import time

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

class Sing2SongDataset:
    def __init__(self, cfg, dataset, is_valid=False):
        # BaseOnlineDataset.__init__(self, cfg, dataset, is_valid=is_valid)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Sing2SongDataset.__init__()")
        self.cfg = cfg
        self.device = self.cfg.audiocraft.device
        self.sample_rate = self.cfg.preprocess.sample_rate
        
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()
        num_samples = cfg.train.valid_num_samples if is_valid else cfg.train.train_num_samples
        if num_samples != -1:
            random.shuffle(self.metadata)
            self.metadata = self.metadata[:min(num_samples, len(self.metadata))]
        
        """
        metadata: a list contains dict, each dict contains:
            dataset(str): dataset name
            action(str): action
            category(str): stem category to generate
            ref_wav(str): reference wav path(for example vocal)
            self_wav(str): self wav path(for example accompaniment, music want to generate)
        """

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Dataset {dataset} loaded, {len(self.metadata)} samples")
        self.metadata_order_by_song = None
        if "stem_dir" in self.metadata[0]:
            self.metadata_order_by_song = {}
            for utt_info in self.metadata:
                song_name = utt_info["stem_dir"]
                if song_name not in self.metadata_order_by_song:
                    self.metadata_order_by_song[song_name] = []
                self.metadata_order_by_song[song_name].append(utt_info)
            
            self.current_song_samples = []  # 存储当前歌曲生成的所有样本
            self.current_song_index = 0  # 当前歌曲的索引
            self.passed_index = 0  # 当前歌曲已经生成的样本数
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Dataset {dataset} loaded, {len(self.metadata_order_by_song)} songs")
            self.load_next_song()  # 加载第一首歌
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] self.load_next_song()")
    
    def load_next_song(self):
        if self.current_song_index < len(self.metadata_order_by_song):
            self.current_song_samples = self.process_song(list(self.metadata_order_by_song.keys())[self.current_song_index])
            self.current_song_index += 1
        else:
            self.current_song_samples = None
    
    # Added logic for stem dataset
    '''
        {
            "dataset": "moisesdb",
            "action": "remove",
            "category": "vocal_lead",
            "ref_wav": [
                "vocal_lead",
                "drum"
            ],
            "self_wav": [
                "drum"
            ],
            "stem_dir": "/nvme/data/zja/stem_v1/wav/Sunny SIde Up_Ding Dong Bell",
            "start": 32.0
        }
    '''
    def process_song(self, song):
        song_stems = {}
        for wav in os.listdir(song):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Processing {wav} in {song}")
            if wav.endswith(".wav"):
                self_wav_path = os.path.join(song, wav)
                self_wav, sr = audio_read(self_wav_path)
                self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
                song_stems[wav[:-4]] = self_wav
        self.song_stems = song_stems
        song_metas = self.metadata_order_by_song[song]
        return song_metas

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata

    def __getitem__(self, index):
        if self.metadata_order_by_song is not None:
            # print(index, self.passed_index, len(self.current_song_samples))
            if index - self.passed_index >= len(self.current_song_samples):
                self.passed_index += len(self.current_song_samples)
                self.load_next_song()
                return self.__getitem__(index)
            utt_info = self.current_song_samples[index - self.passed_index]
        else:
            utt_info = self.metadata[index]

        text_condition = {}
        wav_condition = {}
        wav_path = {}
        
        if "stem_dir" in utt_info:
            # ref_wav
            ref_wav = None
            for ref_wav_name in utt_info["ref_wav"]:
                ref_wav_tmp = self.song_stems[ref_wav_name][:, int(utt_info["start"] * self.sample_rate):int((utt_info["start"] + self.cfg.preprocess.segment_duration) * self.sample_rate)]
                if ref_wav_tmp.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
                    ref_wav_tmp = torch.cat([ref_wav_tmp, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - ref_wav_tmp.shape[-1]], dtype=ref_wav_tmp.dtype, device=ref_wav_tmp.device)], dim=-1)
                ref_wav = ref_wav_tmp if ref_wav is None else ref_wav + ref_wav_tmp
            ref_wav.to(self.device)
            wav_condition["ref_wav"] = ref_wav
            wav_path["ref_wav"] = utt_info["stem_dir"]
            # self_wav
            self_wav = None
            for self_wav_name in utt_info["self_wav"]:
                self_wav_tmp = self.song_stems[self_wav_name][:, int(utt_info["start"] * self.sample_rate):int((utt_info["start"] + self.cfg.preprocess.segment_duration) * self.sample_rate)]
                if self_wav_tmp.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
                    self_wav_tmp = torch.cat([self_wav_tmp, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - self_wav_tmp.shape[-1]], dtype=self_wav_tmp.dtype, device=self_wav_tmp.device)], dim=-1)
                self_wav = self_wav_tmp if self_wav is None else self_wav + self_wav_tmp
            self_wav.to(self.device)
            wav_condition["self_wav"] = self_wav
            wav_path["self_wav"] = utt_info["stem_dir"]
            # TODO: preprocess
            action = utt_info["action"]
            text_condition["action"] = action
            category = utt_info["category"]
            text_condition["category"] = category
            condition_info = ConditionInfo(text_condition, wav_condition, wav_path, self.device, self.sample_rate)
            return self_wav, ref_wav, condition_info
        # self_wav
        self_wav, sr = audio_read(utt_info["self_wav"])
        self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
        # for wavs pad to self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate
        if self_wav.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
            self_wav = torch.cat([self_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - self_wav.shape[-1]], dtype=self_wav.dtype, device=self_wav.device)], dim=-1)
        wav_condition["self_wav"] = self_wav
        wav_path["self_wav"] = utt_info["self_wav"]

        
        # TODO: preprocess
        action = utt_info["action"]
        text_condition["action"] = action
        category = utt_info["category"]
        text_condition["category"] = category
        
        # ref_wav
        # if self.cfg.preprocess.use_ref_wav:
        ref_wav, sr = audio_read(utt_info["ref_wav"])
        ref_wav = convert_audio(ref_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
        if ref_wav.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
            ref_wav = torch.cat([ref_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - ref_wav.shape[-1]], dtype=ref_wav.dtype, device=ref_wav.device)], dim=-1)
        wav_condition["ref_wav"] = ref_wav
        wav_path["ref_wav"] = utt_info["ref_wav"]

        condition_info = ConditionInfo(text_condition, wav_condition, wav_path, self.device, self.sample_rate)
        
        return self_wav, ref_wav, condition_info

    def __len__(self):
        return len(self.metadata)

class Sing2SongCollator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        wavs = torch.stack([x[0] for x in batch])
        ref_wavs = torch.stack([x[1] for x in batch])
        condition_infos = [x[2] for x in batch]
        return wavs, ref_wavs, condition_infos