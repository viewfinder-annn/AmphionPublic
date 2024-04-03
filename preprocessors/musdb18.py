# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torchaudio
import librosa
import json
import tqdm
import itertools

def check_audio_silence_ratio(audio_tensor, silence_threshold=-20.0, frame_length=1024, hop_length=512):
    # print(audio_tensor.shape)
    # to mono
    y = audio_tensor.mean(0, keepdim=False)
    # print(y.shape)
    y = y.numpy()
    # 计算每一帧的短时能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 将能量转换为分贝
    rms_db = librosa.power_to_db(rms, ref=1.0)
    
    # 判断每一帧是否为静音
    silence_frames = (rms_db < silence_threshold)
    
    # 计算静音帧和非静音帧的数量
    non_silence_fraction = 1 - sum(silence_frames) / len(rms_db)
    
    return non_silence_fraction

# TODO: duration & trainfile specification
def main(dataset, output_path, dataset_path):
    # crop every audio to 10s
    save_dir = os.path.join(output_path, dataset)
    wav_dir = os.path.join(save_dir, "wavs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    
    informations = []
    train_dir = os.path.join(dataset_path, "train")
    os.makedirs(train_dir, exist_ok=True)
    for song_dir in tqdm.tqdm(os.listdir(train_dir)):
        song_path = os.path.join(train_dir, song_dir)
        if not os.path.exists(os.path.join(song_path, "vocals.wav")):
            continue
        vocal_audio, vocal_sr = torchaudio.load(os.path.join(song_path, "vocals.wav"))
        bass_audio, bass_sr = torchaudio.load(os.path.join(song_path, "bass.wav"))
        drums_audio, drums_sr = torchaudio.load(os.path.join(song_path, "drums.wav"))
        other_audio, other_sr = torchaudio.load(os.path.join(song_path, "other.wav"))
        stems_all = {
            "vocal": vocal_audio,
            "bass": bass_audio,
            "drums": drums_audio,
            "other": other_audio
        }
        # 裁剪10s的音频 2s overlap
        for i in range(0, vocal_audio.shape[1], 8*vocal_sr):
            if i + 2*vocal_sr > vocal_audio.shape[1]:
                continue
            stems = {}
            for stem in stems_all.keys():
                if check_audio_silence_ratio(stems_all[stem][:, i:i+10*vocal_sr]) > 0.4:
                    stems[stem] = stems_all[stem]
            # print(i//vocal_sr, stems.keys())
            if len(stems) < 2:
                continue
            # add pairs
            for comb in range(1, len(stems)):
                # select stem in stems, in combination
                for input_stem in itertools.combinations(stems.keys(), comb):
                    input_audio = torch.stack([stems[stem][:, i:i+10*vocal_sr] for stem in input_stem]).sum(0)
                    input_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{'_'.join(input_stem)}.wav")
                    if not os.path.exists(input_wav):
                        torchaudio.save(input_wav, input_audio, vocal_sr)
                    output_list = set(stems.keys()) - set(input_stem)
                    for output_stem in output_list:
                        output_audio = stems[output_stem][:, i:i+10*vocal_sr]
                        output_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{output_stem}.wav")
                        if not os.path.exists(output_wav):
                            torchaudio.save(output_wav, output_audio, vocal_sr)
                        informations.append(
                            {
                                "dataset": dataset,
                                "action": "add",
                                "category": output_stem,
                                "ref_wav": os.path.abspath(input_wav),
                                "self_wav": os.path.abspath(output_wav)
                            }
                        )
            # extract & remove pairs
            for comb in range(2, len(stems)+1):
                # select stem in stems, in combination
                for input_stem in itertools.combinations(stems.keys(), comb):
                    input_audio = torch.stack([stems[stem][:, i:i+10*vocal_sr] for stem in input_stem]).sum(0)
                    input_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{'_'.join(input_stem)}.wav")
                    if not os.path.exists(input_wav):
                        torchaudio.save(input_wav, input_audio, vocal_sr)
                    for output_stem in input_stem:
                        # extract
                        output_audio = stems[output_stem][:, i:i+10*vocal_sr]
                        output_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{output_stem}.wav")
                        if not os.path.exists(output_wav):
                            torchaudio.save(output_wav, output_audio, vocal_sr)
                        informations.append(
                            {
                                "dataset": dataset,
                                "action": "extract",
                                "category": output_stem,
                                "ref_wav": os.path.abspath(input_wav),
                                "self_wav": os.path.abspath(output_wav)
                            }
                        )
                        # remove
                        remain_stems = set(input_stem) - set([output_stem])
                        output_audio = torch.stack([stems[stem][:, i:i+10*vocal_sr] for stem in remain_stems]).sum(0)
                        output_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{'_'.join(remain_stems)}.wav")
                        if not os.path.exists(output_wav):
                            torchaudio.save(output_wav, output_audio, vocal_sr)
                        informations.append(
                            {
                                "dataset": dataset,
                                "action": "remove",
                                "category": output_stem,
                                "ref_wav": os.path.abspath(input_wav),
                                "self_wav": os.path.abspath(output_wav)
                            }
                        )
    with open(os.path.join(save_dir, "train.json"), "w") as f:
        json.dump(informations, f, indent=4, ensure_ascii=False)
    
    informations = []
    valid_dir = os.path.join(dataset_path, "test")
    os.makedirs(valid_dir, exist_ok=True)
    for song_dir in tqdm.tqdm(os.listdir(valid_dir)):
        song_path = os.path.join(valid_dir, song_dir)
        if not os.path.exists(os.path.join(song_path, "vocals.wav")):
            continue
        vocal_audio, vocal_sr = torchaudio.load(os.path.join(song_path, "vocals.wav"))
        bass_audio, bass_sr = torchaudio.load(os.path.join(song_path, "bass.wav"))
        drums_audio, drums_sr = torchaudio.load(os.path.join(song_path, "drums.wav"))
        other_audio, other_sr = torchaudio.load(os.path.join(song_path, "other.wav"))
        stems_all = {
            "vocal": vocal_audio,
            "bass": bass_audio,
            "drums": drums_audio,
            "other": other_audio
        }
        # 裁剪10s的音频 2s overlap
        for i in range(0, vocal_audio.shape[1], 8*vocal_sr):
            if i + 2*vocal_sr > vocal_audio.shape[1]:
                continue
            stems = {}
            for stem in stems_all.keys():
                if check_audio_silence_ratio(stems_all[stem][:, i:i+10*vocal_sr]) > 0.4:
                    stems[stem] = stems_all[stem]
            # print(i//vocal_sr, stems.keys())
            if len(stems) < 2:
                continue
            # add pairs
            for comb in range(1, len(stems)):
                # select stem in stems, in combination
                for input_stem in itertools.combinations(stems.keys(), comb):
                    input_audio = torch.stack([stems[stem][:, i:i+10*vocal_sr] for stem in input_stem]).sum(0)
                    input_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{'_'.join(input_stem)}.wav")
                    if not os.path.exists(input_wav):
                        torchaudio.save(input_wav, input_audio, vocal_sr)
                    output_list = set(stems.keys()) - set(input_stem)
                    for output_stem in output_list:
                        output_audio = stems[output_stem][:, i:i+10*vocal_sr]
                        output_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{output_stem}.wav")
                        if not os.path.exists(output_wav):
                            torchaudio.save(output_wav, output_audio, vocal_sr)
                        informations.append(
                            {
                                "dataset": dataset,
                                "action": "add",
                                "category": output_stem,
                                "ref_wav": os.path.abspath(input_wav),
                                "self_wav": os.path.abspath(output_wav)
                            }
                        )
            # extract & remove pairs
            for comb in range(2, len(stems)+1):
                # select stem in stems, in combination
                for input_stem in itertools.combinations(stems.keys(), comb):
                    input_audio = torch.stack([stems[stem][:, i:i+10*vocal_sr] for stem in input_stem]).sum(0)
                    input_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{'_'.join(input_stem)}.wav")
                    if not os.path.exists(input_wav):
                        torchaudio.save(input_wav, input_audio, vocal_sr)
                    for output_stem in input_stem:
                        # extract
                        output_audio = stems[output_stem][:, i:i+10*vocal_sr]
                        output_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{output_stem}.wav")
                        if not os.path.exists(output_wav):
                            torchaudio.save(output_wav, output_audio, vocal_sr)
                        informations.append(
                            {
                                "dataset": dataset,
                                "action": "extract",
                                "category": output_stem,
                                "ref_wav": os.path.abspath(input_wav),
                                "self_wav": os.path.abspath(output_wav)
                            }
                        )
                        # remove
                        remain_stems = set(input_stem) - set([output_stem])
                        output_audio = torch.stack([stems[stem][:, i:i+10*vocal_sr] for stem in remain_stems]).sum(0)
                        output_wav = os.path.join(wav_dir, f"{song_dir}_{i//vocal_sr}_{'_'.join(remain_stems)}.wav")
                        if not os.path.exists(output_wav):
                            torchaudio.save(output_wav, output_audio, vocal_sr)
                        informations.append(
                            {
                                "dataset": dataset,
                                "action": "remove",
                                "category": output_stem,
                                "ref_wav": os.path.abspath(input_wav),
                                "self_wav": os.path.abspath(output_wav)
                            }
                        )
    with open(os.path.join(save_dir, "test.json"), "w") as f:
        json.dump(informations, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main("musdb18", "/nvme/amphion/zhfang/zja/_datase_utils/preprocessor", "/nvme/amphion/zhfang/zja/_dataset/musdb18")
    # main("musdb18", "/home/wangyuancheng.p/zja/Amphion/data", "/nvme/amphion/zhfang/zja/_dataset/musdb18")