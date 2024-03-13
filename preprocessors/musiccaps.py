# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torchaudio
import pandas
import json
import tqdm

def main(dataset, output_path, dataset_path):
    # crop every audio to 10s
    save_dir = os.path.join(output_path, dataset)
    wav_dir = os.path.join(save_dir, "wavs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    
    meta_dir = os.path.join(dataset_path, "musiccaps-public.csv")

    meta = pandas.read_csv(meta_dir)
    informations = []
    
    datas = [data for data in os.listdir(dataset_path) if os.path.splitext(data)[1] == ".wav"]
    for data in tqdm.tqdm(datas):
        name, ext = os.path.splitext(data)
        if name in meta["ytid"].values:
            audio_file = os.path.join(dataset_path, data)
            row = meta[meta["ytid"] == name]
            # pick the first one
            row = row.iloc[0].to_dict()
            
            audio_dst = os.path.join(wav_dir, data)
            if not os.path.exists(audio_dst):
                # keep the first 10s, pad if shorter
                y, sr = torchaudio.load(audio_file)
                y = y[:, :10*sr]
                if y.shape[1] < 10*sr:
                    y = torch.nn.functional.pad(y, (0, 10*sr-y.shape[1]))
                torchaudio.save(audio_dst, y, sr)
                
            informations.append(
                {
                    "dataset": dataset,
                    "caption": row["caption"],
                    "self_wav": os.path.abspath(audio_dst),
                }
            )
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(informations, f, indent=4, ensure_ascii=False)