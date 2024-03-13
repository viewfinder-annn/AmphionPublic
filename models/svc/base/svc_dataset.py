# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import os
import numpy as np
import librosa

from utils.data_utils import *
from processors.acoustic_extractor import cal_normalized_mel, load_mel_extrema
from processors.content_extractor import (
    ContentvecExtractor,
    WhisperExtractor,
    WenetExtractor,
)
from models.base.base_dataset import (
    BaseOfflineDataset,
    BaseOfflineCollator,
    BaseOnlineDataset,
    BaseOnlineCollator,
)
from models.base.new_dataset import BaseTestDataset

EPS = 1.0e-12


class SVCOfflineDataset(BaseOfflineDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        BaseOfflineDataset.__init__(self, cfg, dataset, is_valid=is_valid)

        cfg = self.cfg

        if cfg.model.condition_encoder.use_whisper:
            self.whisper_aligner = WhisperExtractor(self.cfg)
            self.utt2whisper_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.whisper_dir
            )

        if cfg.model.condition_encoder.use_contentvec:
            self.contentvec_aligner = ContentvecExtractor(self.cfg)
            self.utt2contentVec_path = load_content_feature_path(
                self.metadata,
                cfg.preprocess.processed_dir,
                cfg.preprocess.contentvec_dir,
            )

        if cfg.model.condition_encoder.use_mert:
            self.utt2mert_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.mert_dir
            )
        if cfg.model.condition_encoder.use_wenet:
            self.wenet_aligner = WenetExtractor(self.cfg)
            self.utt2wenet_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.wenet_dir
            )

    def __getitem__(self, index):
        single_feature = BaseOfflineDataset.__getitem__(self, index)

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        if self.cfg.model.condition_encoder.use_whisper:
            assert "target_len" in single_feature.keys()
            aligned_whisper_feat = (
                self.whisper_aligner.offline_resolution_transformation(
                    np.load(self.utt2whisper_path[utt]), single_feature["target_len"]
                )
            )
            single_feature["whisper_feat"] = aligned_whisper_feat

        if self.cfg.model.condition_encoder.use_contentvec:
            assert "target_len" in single_feature.keys()
            aligned_contentvec = (
                self.contentvec_aligner.offline_resolution_transformation(
                    np.load(self.utt2contentVec_path[utt]), single_feature["target_len"]
                )
            )
            single_feature["contentvec_feat"] = aligned_contentvec

        if self.cfg.model.condition_encoder.use_mert:
            assert "target_len" in single_feature.keys()
            aligned_mert_feat = align_content_feature_length(
                np.load(self.utt2mert_path[utt]),
                single_feature["target_len"],
                source_hop=self.cfg.preprocess.mert_hop_size,
            )
            single_feature["mert_feat"] = aligned_mert_feat

        if self.cfg.model.condition_encoder.use_wenet:
            assert "target_len" in single_feature.keys()
            aligned_wenet_feat = self.wenet_aligner.offline_resolution_transformation(
                np.load(self.utt2wenet_path[utt]), single_feature["target_len"]
            )
            single_feature["wenet_feat"] = aligned_wenet_feat

        # print(single_feature.keys())
        # for k, v in single_feature.items():
        #     if type(v) in [torch.Tensor, np.ndarray]:
        #         print(k, v.shape)
        #     else:
        #         print(k, v)
        # exit()

        return self.clip_if_too_long(single_feature)

    def __len__(self):
        return len(self.metadata)

    def random_select(self, feature_seq_len, max_seq_len, ending_ts=2812):
        """
        ending_ts: to avoid invalid whisper features for over 30s audios
            2812 = 30 * 24000 // 256
        """
        ts = max(feature_seq_len - max_seq_len, 0)
        ts = min(ts, ending_ts - max_seq_len)

        start = random.randint(0, ts)
        end = start + max_seq_len
        return start, end

    def clip_if_too_long(self, sample, max_seq_len=512):
        """
        sample :
            {
                'spk_id': (1,),
                'target_len': int
                'mel': (seq_len, dim),
                'frame_pitch': (seq_len,)
                'frame_energy': (seq_len,)
                'content_vector_feat': (seq_len, dim)
            }
        """

        if sample["target_len"] <= max_seq_len:
            return sample

        start, end = self.random_select(sample["target_len"], max_seq_len)
        sample["target_len"] = end - start

        for k in sample.keys():
            if k == "audio":
                # audio should be clipped in hop_size scale
                sample[k] = sample[k][
                    start
                    * self.cfg.preprocess.hop_size : end
                    * self.cfg.preprocess.hop_size
                ]
            elif k == "audio_len":
                sample[k] = (end - start) * self.cfg.preprocess.hop_size
            elif k not in ["spk_id", "target_len"]:
                sample[k] = sample[k][start:end]

        return sample


class SVCOnlineDataset(BaseOnlineDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        super().__init__(cfg, dataset, is_valid=is_valid)

        # Audio pretrained models' sample rates
        self.all_sample_rates = {self.sample_rate}
        if self.cfg.model.condition_encoder.use_whisper:
            self.all_sample_rates.add(self.cfg.preprocess.whisper_sample_rate)
        if self.cfg.model.condition_encoder.use_contentvec:
            self.all_sample_rates.add(self.cfg.preprocess.contentvec_sample_rate)
        if self.cfg.model.condition_encoder.use_wenet:
            self.all_sample_rates.add(self.cfg.preprocess.wenet_sample_rate)

        self.highest_sample_rate = max(list(self.all_sample_rates))

        # The maximum duration (seconds) for one training sample
        self.max_duration = 6.0
        self.max_n_frames = int(self.max_duration * self.highest_sample_rate)

    def random_select(self, wav, duration, wav_path):
        """
        wav: (T,)
        """
        if duration <= self.max_duration:
            return wav

        ts_frame = int((duration - self.max_duration) * self.highest_sample_rate)
        start = random.randint(0, ts_frame)
        end = start + self.max_n_frames

        if (wav[start:end] == 0).all():
            print("*" * 20)
            print("Warning! The wav file {} has a lot of silience.".format(wav_path))

            # There should be at least some frames that are not silience. Then we select them.
            assert (wav != 0).any()
            start = np.where(wav != 0)[0][0]
            end = start + self.max_n_frames

        return wav[start:end]

    def __getitem__(self, index):
        """
        single_feature: dict,
            wav: (T,)
            wav_len: int
            target_len: int
            mask: (n_frames, 1)
            spk_id

            wav_{sr}: (T,)
            wav_{sr}_len: int
        """
        single_feature = dict()

        utt_item = self.metadata[index]
        wav_path = utt_item["Path"]

        ### Use the highest sampling rate to load and randomly select ###
        highest_sr_wav, _ = librosa.load(wav_path, sr=self.highest_sample_rate)
        highest_sr_wav = self.random_select(
            highest_sr_wav, utt_item["Duration"], wav_path
        )

        ### Waveforms under all the sample rates ###
        for sr in self.all_sample_rates:
            # Resample to the required sample rate
            if sr != self.highest_sample_rate:
                wav_sr = librosa.resample(
                    highest_sr_wav, orig_sr=self.highest_sample_rate, target_sr=sr
                )
            else:
                wav_sr = highest_sr_wav

            wav_sr = torch.as_tensor(wav_sr, dtype=torch.float32)
            single_feature["wav_{}".format(sr)] = wav_sr
            single_feature["wav_{}_len".format(sr)] = len(wav_sr)

            # For target sample rate
            if sr == self.sample_rate:
                wav_len = len(wav_sr)
                frame_len = wav_len // self.hop_size

                single_feature["wav"] = wav_sr
                single_feature["wav_len"] = wav_len
                single_feature["target_len"] = frame_len
                single_feature["mask"] = torch.ones(frame_len, 1, dtype=torch.long)

        ### Speaker ID ###
        if self.cfg.preprocess.use_spkid:
            utt = "{}_{}".format(utt_item["Dataset"], utt_item["Uid"])
            single_feature["spk_id"] = torch.tensor(
                [self.spk2id[self.utt2spk[utt]]], dtype=torch.int32
            )

        return single_feature

    def __len__(self):
        return len(self.metadata)


class SVCOfflineCollator(BaseOfflineCollator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch):
        parsed_batch_features = super().__call__(batch)
        return parsed_batch_features


class SVCOnlineCollator(BaseOnlineCollator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch):
        """
        SVCOnlineDataset.__getitem__:
            wav: (T,)
            wav_len: int
            target_len: int
            mask: (n_frames, 1)
            spk_id: (1)

            wav_{sr}: (T,)
            wav_{sr}_len: int

        Returns:
            wav: (B, T), torch.float32
            wav_len: (B), torch.long
            target_len: (B), torch.long
            mask: (B, n_frames, 1), torch.long
            spk_id: (B, 1), torch.int32

            wav_{sr}: (B, T)
            wav_{sr}_len: (B), torch.long
        """
        packed_batch_features = dict()

        for key in batch[0].keys():
            if "_len" in key:
                packed_batch_features[key] = torch.LongTensor([b[key] for b in batch])
            else:
                packed_batch_features[key] = pad_sequence(
                    [b[key] for b in batch], batch_first=True, padding_value=0
                )
        return packed_batch_features


class SVCTestDataset(BaseTestDataset):
    def __init__(self, args, cfg, infer_type):
        BaseTestDataset.__init__(self, args, cfg, infer_type)
        self.metadata = self.get_metadata()

        target_singer = args.target_singer
        self.cfg = cfg
        self.trans_key = args.trans_key
        assert type(target_singer) == str

        self.target_singer = target_singer.split("_")[-1]
        self.target_dataset = target_singer.replace(
            "_{}".format(self.target_singer), ""
        )
        if cfg.preprocess.mel_min_max_norm:
            if self.cfg.preprocess.features_extraction_mode == "online":
                # TODO: Change the hard code

                # Using an empirical mel extrema to normalize
                self.target_mel_extrema = load_mel_extrema(cfg.preprocess, "vctk")
            else:
                self.target_mel_extrema = load_mel_extrema(
                    cfg.preprocess, self.target_dataset
                )

            self.target_mel_extrema = torch.as_tensor(
                self.target_mel_extrema[0]
            ), torch.as_tensor(self.target_mel_extrema[1])

        ######### Load source acoustic features #########
        if cfg.preprocess.use_spkid:
            spk2id_path = os.path.join(args.acoustics_dir, cfg.preprocess.spk2id)
            # utt2sp_path = os.path.join(self.data_root, cfg.preprocess.utt2spk)

            with open(spk2id_path, "r", encoding="utf-8") as f:
                self.spk2id = json.load(f)
            # print("self.spk2id", self.spk2id)

        if cfg.preprocess.use_uv:
            self.utt2uv_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.uv_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

        if cfg.preprocess.use_frame_pitch:
            self.utt2frame_pitch_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.pitch_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

            # Target F0 median
            target_f0_statistics_path = os.path.join(
                cfg.preprocess.processed_dir,
                self.target_dataset,
                cfg.preprocess.pitch_dir,
                "statistics.json",
            )
            self.target_pitch_median = json.load(
                open(target_f0_statistics_path, "r", encoding="utf-8")
            )[f"{self.target_dataset}_{self.target_singer}"]["voiced_positions"][
                "median"
            ]

            # Source F0 median (if infer from file)
            if infer_type == "from_file":
                source_audio_name = cfg.inference.source_audio_name
                source_f0_statistics_path = os.path.join(
                    cfg.preprocess.processed_dir,
                    source_audio_name,
                    cfg.preprocess.pitch_dir,
                    "statistics.json",
                )
                self.source_pitch_median = json.load(
                    open(source_f0_statistics_path, "r", encoding="utf-8")
                )[f"{source_audio_name}_{source_audio_name}"]["voiced_positions"][
                    "median"
                ]
            else:
                self.source_pitch_median = None

        if cfg.preprocess.use_frame_energy:
            self.utt2frame_energy_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.energy_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

        if cfg.preprocess.use_mel:
            self.utt2mel_path = {
                f'{utt_info["Dataset"]}_{utt_info["Uid"]}': os.path.join(
                    cfg.preprocess.processed_dir,
                    utt_info["Dataset"],
                    cfg.preprocess.mel_dir,
                    utt_info["Uid"] + ".npy",
                )
                for utt_info in self.metadata
            }

        ######### Load source content features' path #########
        if cfg.model.condition_encoder.use_whisper:
            self.whisper_aligner = WhisperExtractor(cfg)
            self.utt2whisper_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.whisper_dir
            )

        if cfg.model.condition_encoder.use_contentvec:
            self.contentvec_aligner = ContentvecExtractor(cfg)
            self.utt2contentVec_path = load_content_feature_path(
                self.metadata,
                cfg.preprocess.processed_dir,
                cfg.preprocess.contentvec_dir,
            )

        if cfg.model.condition_encoder.use_mert:
            self.utt2mert_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.mert_dir
            )
        if cfg.model.condition_encoder.use_wenet:
            self.wenet_aligner = WenetExtractor(cfg)
            self.utt2wenet_path = load_content_feature_path(
                self.metadata, cfg.preprocess.processed_dir, cfg.preprocess.wenet_dir
            )

    def __getitem__(self, index):
        single_feature = {}

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        source_dataset = self.metadata[index]["Dataset"]

        if self.cfg.preprocess.use_spkid:
            single_feature["spk_id"] = np.array(
                [self.spk2id[f"{self.target_dataset}_{self.target_singer}"]],
                dtype=np.int32,
            )

        ######### Get Acoustic Features Item #########
        if self.cfg.preprocess.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.preprocess.n_mel  # [n_mels, T]
            if self.cfg.preprocess.use_min_max_norm_mel:
                # mel norm
                mel = cal_normalized_mel(mel, source_dataset, self.cfg.preprocess)

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]
            single_feature["mel"] = mel.T  # [T, n_mels]

        if self.cfg.preprocess.use_frame_pitch:
            frame_pitch_path = self.utt2frame_pitch_path[utt]
            frame_pitch = np.load(frame_pitch_path)

            if self.trans_key:
                try:
                    self.trans_key = int(self.trans_key)
                except:
                    pass
                if type(self.trans_key) == int:
                    frame_pitch = transpose_key(frame_pitch, self.trans_key)
                elif self.trans_key:
                    assert self.target_singer

                    frame_pitch = pitch_shift_to_target(
                        frame_pitch, self.target_pitch_median, self.source_pitch_median
                    )

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)
            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )
            single_feature["frame_pitch"] = aligned_frame_pitch

            if self.cfg.preprocess.use_uv:
                frame_uv_path = self.utt2uv_path[utt]
                frame_uv = np.load(frame_uv_path)
                aligned_frame_uv = align_length(frame_uv, single_feature["target_len"])
                aligned_frame_uv = [
                    0 if frame_uv else 1 for frame_uv in aligned_frame_uv
                ]
                aligned_frame_uv = np.array(aligned_frame_uv)
                single_feature["frame_uv"] = aligned_frame_uv

        if self.cfg.preprocess.use_frame_energy:
            frame_energy_path = self.utt2frame_energy_path[utt]
            frame_energy = np.load(frame_energy_path)
            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_energy)
            aligned_frame_energy = align_length(
                frame_energy, single_feature["target_len"]
            )
            single_feature["frame_energy"] = aligned_frame_energy

        ######### Get Content Features Item #########
        if self.cfg.model.condition_encoder.use_whisper:
            assert "target_len" in single_feature.keys()
            aligned_whisper_feat = (
                self.whisper_aligner.offline_resolution_transformation(
                    np.load(self.utt2whisper_path[utt]), single_feature["target_len"]
                )
            )
            single_feature["whisper_feat"] = aligned_whisper_feat

        if self.cfg.model.condition_encoder.use_contentvec:
            assert "target_len" in single_feature.keys()
            aligned_contentvec = (
                self.contentvec_aligner.offline_resolution_transformation(
                    np.load(self.utt2contentVec_path[utt]), single_feature["target_len"]
                )
            )
            single_feature["contentvec_feat"] = aligned_contentvec

        if self.cfg.model.condition_encoder.use_mert:
            assert "target_len" in single_feature.keys()
            aligned_mert_feat = align_content_feature_length(
                np.load(self.utt2mert_path[utt]),
                single_feature["target_len"],
                source_hop=self.cfg.preprocess.mert_hop_size,
            )
            single_feature["mert_feat"] = aligned_mert_feat

        if self.cfg.model.condition_encoder.use_wenet:
            assert "target_len" in single_feature.keys()
            aligned_wenet_feat = self.wenet_aligner.offline_resolution_transformation(
                np.load(self.utt2wenet_path[utt]), single_feature["target_len"]
            )
            single_feature["wenet_feat"] = aligned_wenet_feat

        return single_feature

    def __len__(self):
        return len(self.metadata)


class SVCTestCollator:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, T, n_mels]
        # frame_pitch, frame_energy: [1, T]
        # target_len: [1]
        # spk_id: [b, 1]
        # mask: [b, T, 1]

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
