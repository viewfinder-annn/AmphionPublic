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

def check_audio_silence_ratio(audio_tensor, silence_threshold=-30.0, frame_length=1024, hop_length=512):
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
def main(dataset, output_path, dataset_path, no_drums=False):
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
        # 裁剪10s的音频, 并保证裁剪的部分里vocal不是没有声音的
        for i in range(0, vocal_audio.shape[1], 5*vocal_sr):
            if i + 5*vocal_sr > vocal_audio.shape[1]:
                continue
            # print(check_audio_silence_ratio(vocal_audio[:, i:i+10*vocal_sr]), check_audio_silence_ratio(drums_audio[:, i:i+10*vocal_sr]))
            if check_audio_silence_ratio(vocal_audio[:, i:i+10*vocal_sr]) > 0.6 and check_audio_silence_ratio(drums_audio[:, i:i+10*vocal_sr]) > 0.4:
                vocal_dst = os.path.join(wav_dir, f"{song_dir}_vocal_{i//vocal_sr}.wav")
                instrumental = os.path.join(wav_dir, f"{song_dir}_instrumental_{i//vocal_sr}.wav")
                instrumental_no_drums = os.path.join(wav_dir, f"{song_dir}_instrumental_no_drums_{i//vocal_sr}.wav")
                # mixture = os.path.join(wav_dir, f"{song_dir}_mixture_{i//vocal_sr}.wav")
                if not os.path.exists(vocal_dst):
                    torchaudio.save(vocal_dst, vocal_audio[:, i:i+10*vocal_sr], vocal_sr)
                if not os.path.exists(instrumental_no_drums):
                    torchaudio.save(instrumental_no_drums, torch.stack([bass_audio[:, i:i+10*vocal_sr], other_audio[:, i:i+10*vocal_sr]]).sum(0), bass_sr)
                informations.append(
                    {
                        "dataset": dataset,
                        "caption": "Generate accompaniment without drums from vocal",
                        "ref_wav": os.path.abspath(vocal_dst),
                        "self_wav": os.path.abspath(instrumental_no_drums)
                    }
                )
                if not no_drums:
                    if not os.path.exists(vocal_dst):
                        torchaudio.save(vocal_dst, vocal_audio[:, i:i+10*vocal_sr], vocal_sr)
                    if not os.path.exists(instrumental):
                        torchaudio.save(instrumental, torch.stack([bass_audio[:, i:i+10*vocal_sr], drums_audio[:, i:i+10*vocal_sr], other_audio[:, i:i+10*vocal_sr]]).sum(0), bass_sr)
                    informations.append(
                        {
                            "dataset": dataset,
                            "caption": "Generate accompaniment from vocal",
                            "ref_wav": os.path.abspath(vocal_dst),
                            "self_wav": os.path.abspath(instrumental)
                        }
                    )
                # if not os.path.exists(mixture):
                #     torchaudio.save(mixture, torch.stack([bass_audio[:, i:i+10*vocal_sr], drums_audio[:, i:i+10*vocal_sr], other_audio[:, i:i+10*vocal_sr], vocal_audio[:, i:i+10*vocal_sr]]).sum(0), bass_sr)  
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
        # 裁剪10s的音频, 并保证裁剪的部分里vocal不是没有声音的
        for i in range(0, vocal_audio.shape[1], 5*vocal_sr):
            if i + 5*vocal_sr > vocal_audio.shape[1]:
                continue
            if check_audio_silence_ratio(vocal_audio[:, i:i+10*vocal_sr]) > 0.6 and check_audio_silence_ratio(drums_audio[:, i:i+10*vocal_sr]) > 0.4:
                vocal_dst = os.path.join(wav_dir, f"{song_dir}_vocal_{i//vocal_sr}.wav")
                instrumental = os.path.join(wav_dir, f"{song_dir}_instrumental_{i//vocal_sr}.wav")
                instrumental_no_drums = os.path.join(wav_dir, f"{song_dir}_instrumental_no_drums_{i//vocal_sr}.wav")
                # mixture = os.path.join(wav_dir, f"{song_dir}_mixture_{i//vocal_sr}.wav")
                if not os.path.exists(vocal_dst):
                    torchaudio.save(vocal_dst, vocal_audio[:, i:i+10*vocal_sr], vocal_sr)
                if not os.path.exists(instrumental_no_drums):
                    torchaudio.save(instrumental_no_drums, torch.stack([bass_audio[:, i:i+10*vocal_sr], other_audio[:, i:i+10*vocal_sr]]).sum(0), bass_sr)
                informations.append(
                    {
                        "dataset": dataset,
                        "caption": "Generate accompaniment without drums from vocal",
                        "ref_wav": os.path.abspath(vocal_dst),
                        "self_wav": os.path.abspath(instrumental_no_drums)
                    }
                )
                if not no_drums:
                    if not os.path.exists(vocal_dst):
                        torchaudio.save(vocal_dst, vocal_audio[:, i:i+10*vocal_sr], vocal_sr)
                    if not os.path.exists(instrumental):
                        torchaudio.save(instrumental, torch.stack([bass_audio[:, i:i+10*vocal_sr], drums_audio[:, i:i+10*vocal_sr], other_audio[:, i:i+10*vocal_sr]]).sum(0), bass_sr)
                    informations.append(
                        {
                            "dataset": dataset,
                            "caption": "Generate accompaniment from vocal",
                            "ref_wav": os.path.abspath(vocal_dst),
                            "self_wav": os.path.abspath(instrumental)
                        }
                    )
                # if not os.path.exists(mixture):
                #     torchaudio.save(mixture, torch.stack([bass_audio[:, i:i+10*vocal_sr], drums_audio[:, i:i+10*vocal_sr], other_audio[:, i:i+10*vocal_sr], vocal_audio[:, i:i+10*vocal_sr]]).sum(0), bass_sr)         
    with open(os.path.join(save_dir, "test.json"), "w") as f:
        json.dump(informations, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main("musdb18_no_drums", "/nvme/amphion/zhfang/zja/_datase_utils/preprocessor", "/nvme/amphion/zhfang/zja/_dataset/musdb18", no_drums=True)
    # main("musdb18", "/nvme/amphion/zhfang/zja/_datase_utils/preprocessor", "/nvme/amphion/zhfang/zja/_dataset/musdb18")
    # main("musdb18", "/home/wangyuancheng.p/zja/Amphion/data", "/nvme/amphion/zhfang/zja/_dataset/musdb18")