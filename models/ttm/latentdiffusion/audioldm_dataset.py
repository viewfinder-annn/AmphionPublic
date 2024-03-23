# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import json
from utils.data_utils import *
from torch.nn.utils.rnn import pad_sequence

from models.base.base_dataset import (
    BaseOnlineDataset,
    BaseOnlineCollator,
    BaseTestDataset,
    BaseTestCollator,
)
import librosa

from modules.audiocraft.data.audio import audio_read
from modules.audiocraft.data.audio_utils import convert_audio
from utils.mel import extract_mel_features
from transformers import AutoTokenizer

class AudioLDMDataset:
    def __init__(self, cfg, dataset, is_valid=False):
        # BaseOnlineDataset.__init__(self, cfg, dataset, is_valid=is_valid)

        self.cfg = cfg
        # self.device = self.cfg.audiocraft.device
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
        wav_path = {}
        
        single_feature = {}
        
        # self_wav
        self_wav, sr = audio_read(utt_info["self_wav"])
        self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
        # for wavs pad to self.cfg.preprocess.segment_duration * self.cfg.preprocess.sample_rate
        if self_wav.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
            self_wav = torch.cat([self_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - self_wav.shape[-1]], dtype=self_wav.dtype, device=self_wav.device)], dim=-1)
        wav_condition["self_wav"] = self_wav
        wav_path["self_wav"] = utt_info["self_wav"]
        single_feature["wav"] = self_wav
        single_feature["mel"] = extract_mel_features(self_wav, self.cfg.preprocess)

        # caption
        if self.cfg.preprocess.use_caption:
            caption = utt_info["caption"]
            text_condition["description"] = caption
            single_feature["caption"] = caption

        # ref_wav
        if self.cfg.preprocess.use_ref_wav:
            ref_wav, sr = audio_read(utt_info["ref_wav"])
            ref_wav = convert_audio(ref_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
            if ref_wav.shape[-1] < self.cfg.preprocess.segment_duration * self.sample_rate:
                ref_wav = torch.cat([ref_wav, torch.zeros([self.cfg.preprocess.audio_channels, self.cfg.preprocess.segment_duration * self.sample_rate - ref_wav.shape[-1]], dtype=ref_wav.dtype, device=ref_wav.device)], dim=-1)
            wav_condition["ref_wav"] = ref_wav
            wav_path["ref_wav"] = utt_info["ref_wav"]
            single_feature["ref_wav"] = ref_wav
        
        return single_feature

    def __len__(self):
        return len(self.metadata)

class AudioLDMCollator:
    def __init__(self, cfg):

        self.tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)

    def __call__(self, batch):
        # mel: (B, n_mels, T)
        # wav (option): (B, T)
        # text_input_ids: (B, L)
        # text_attention_mask: (B, L)

        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "mel":
                packed_batch_features["mel"] = torch.from_numpy(
                    np.array([b["mel"][:, :624] for b in batch])
                )

            if key == "wav":
                values = [b[key] for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

            if key == "caption":
                captions = [b[key] for b in batch]
                text_input = self.tokenizer(
                    captions, return_tensors="pt", truncation=True, padding="longest"
                )
                text_input_ids = text_input["input_ids"]
                text_attention_mask = text_input["attention_mask"]

                packed_batch_features["text_input_ids"] = text_input_ids
                packed_batch_features["text_attention_mask"] = text_attention_mask

        return packed_batch_features