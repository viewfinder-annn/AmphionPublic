# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.base.base_dataset import (
    BaseOfflineCollator,
    BaseOfflineDataset,
    BaseTestDataset,
    BaseTestCollator,
)
import librosa
from modules.audiocraft.data.audio import audio_read
from modules.audiocraft.data.audio_utils import convert_audio

from demucs import pretrained
from demucs.apply import apply_model
from models.ttm.latentdiffusion.mel import mel_spectrogram

class AutoencoderKLDataset:
    def __init__(self, cfg, dataset, is_valid=False):
        # BaseOfflineDataset.__init__(self, cfg, dataset, is_valid=is_valid)

        self.cfg = cfg
        self.sample_rate = self.cfg.preprocess.sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        if "lyric" in self.metadata[0]:
            self.demucs = pretrained.get_model('htdemucs')

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata

    def __getitem__(self, index):
        # melspec: (n_mels, T)
        # wav: (T,)

        single_feature = {}
        # print(self.metadata[index])

        utt_info = self.metadata[index]
        # dataset = utt_info["dataset"]
        # uid = os.path.basename(utt_info["self_wav"])
        # utt = "{}_{}".format(dataset, uid)
        
        # sing2song
        if "lyric" in utt_info:
            self_wav, sr = audio_read(utt_info["self_wav"], seek_time=utt_info["start"], duration=self.cfg.preprocess.segment_duration, pad=True)
            # 0514 added compare
            self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
            single_feature["self_wav"] = self_wav
            
            # # Original
            # if self_wav.dim() == 1:
            #     self_wav = self_wav.unsqueeze(0)
            # res = apply_model(self.demucs, self_wav.unsqueeze(0), device=self.device) # [1,4,2,duration*sr] 
            # res.squeeze_(0)
            # ref_wav = res[-1]
            # ref_wav = convert_audio(ref_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
            # self_wav = res[:-1].sum(0)
            # self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
            
            # # # add noise to ref_wav(vocal)
            # # noise = torch.randn_like(self_wav)
            # # # -45dB
            # # noise = noise * 10**(-45/20)
            # # ref_wav = ref_wav + noise
            
            # self_wav = random.choice([self_wav, ref_wav])
            # single_feature["self_wav"] = self_wav
        else:
            self_wav, sr = audio_read(utt_info["self_wav"])
            self_wav = convert_audio(self_wav, sr, self.sample_rate, self.cfg.preprocess.audio_channels)
            

        if self.cfg.preprocess.use_mel:
            single_feature["mel"] = mel_spectrogram(
                self_wav,
                n_fft=self.cfg.preprocess.n_fft,
                num_mels=self.cfg.preprocess.n_mel,
                sampling_rate=self.sample_rate,
                hop_size=self.cfg.preprocess.hop_size,
                win_size=self.cfg.preprocess.win_size,
                fmin=self.cfg.preprocess.fmin,
                fmax=self.cfg.preprocess.fmax,
            ) # [1, 80, 625]
            single_feature["mel"].squeeze_(0)
            # print(single_feature["mel"].shape) # [80, 625]

        return single_feature

    def __len__(self):
        return len(self.metadata)

    def __len__(self):
        return len(self.metadata)


class AutoencoderKLCollator(BaseOfflineCollator):
    def __init__(self, cfg):
        BaseOfflineCollator.__init__(self, cfg)

    def __call__(self, batch):
        # mel: (B, n_mels, T)
        # wav (option): (B, T)

        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "mel":
                packed_batch_features["mel"] = torch.stack([b[key][:, :624] for b in batch], dim=0)

            if key == "wav":
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features


class AutoencoderKLTestDataset(BaseTestDataset): ...


class AutoencoderKLTestCollator(BaseTestCollator): ...

if __name__ == "__main__":
    from utils.util import load_config
    cfg = load_config("/home/wangyuancheng.p/zja/Amphion/egs/ttm/latentdiffusion/autoencoderkl/exp_sing2song_mel.json")
    dataset = AutoencoderKLDataset(cfg, "musdb18")
    for i in range(len(dataset)):
        print(dataset[i])
        break