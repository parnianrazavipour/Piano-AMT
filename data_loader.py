from pathlib import Path
from typing import Optional
import os
from audidata.io.audio import load, random_start_time

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import ToMono
from audidata.transforms.midi import PianoRoll
from audidata.io.midi import read_single_track_midi
from audidata.collate.base import collate_list_fn
import yaml


default_collate_fn_map.update({list: collate_list_fn})


class MAESTRO(Dataset):
    r"""MAESTRO [1] is a dataset containing 199 hours of 1,276 audio files and 
    aligned MIDI files captured by Yamaha Disklaiver. Audios are sampled at 44,100 Hz. 
    After decompression, the dataset is 131 GB.

    [1] C. Hawthorne, et al., Enabling Factorized Piano Music Modeling and 
    Generation with the MAESTRO Dataset, ICLR, 2019

    The dataset looks like:

        dataset_root (131 GB)
        ├── 2004 (132 songs, wav + flac + midi + tsv)
        ├── 2006 (115 songs, wav + flac + midi + tsv)
        ├── 2008 (147 songs, wav + flac + midi + tsv)
        ├── 2009 (125 songs, wav + flac + midi + tsv)
        ├── 2011 (163 songs, wav + flac + midi + tsv)
        ├── 2013 (127 songs, wav + flac + midi + tsv)
        ├── 2014 (105 songs, wav + flac + midi + tsv)
        ├── 2015 (129 songs, wav + flac + midi + tsv)
        ├── 2017 (140 songs, wav + flac + midi + tsv)
        ├── 2018 (93 songs, wav + flac + midi + tsv)
        ├── LICENSE
        ├── maestro-v3.0.0.csv
        ├── maestro-v3.0.0.json
        └── README
    """

    url = "https://magenta.tensorflow.org/datasets/maestro"

    duration = 717232.49  # Dataset duration (s), 199 hours, including training, 
    # validation, and testing.

    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        crop: Optional[callable] = RandomCrop(clip_duration=10., end_pad=9.9),
        transform: Optional[callable] = ToMono(),
        target: bool = True,
        extend_pedal: bool = True,
        target_transform: Optional[callable] = PianoRoll(fps=100, pitches_num=128),
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.clip_duration = 10
        self.clip_samples = round(self.clip_duration * self.sr)
        self.target = target
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform

        meta_csv = Path(self.root, "maestro-v3.0.0.csv")

        self.meta_dict = self.load_meta(meta_csv)
        
    def __getitem__(self, index: int) -> dict:

        audio_path = Path(self.root, self.meta_dict["audio_name"][index])
        midi_path = Path(self.root, self.meta_dict["midi_name"][index]) 
        duration = self.meta_dict["duration"][index]

        full_data = {
            "dataset_name": "MAESTRO V3.0.0",
            "audio_path": str(audio_path),
        }

        start_time = random_start_time(audio_path)

        # Load audio
        audio_data = self.load_audio(path=audio_path , start_time=start_time)
        full_data.update(audio_data)
        
        # Load target
        if self.target:
            target_data = self.load_target(
                midi_path=midi_path, 
                start_time=audio_data["start_time"],
                clip_duration=audio_data["duration"]
            )
            full_data.update(target_data)

        return full_data

    def __len__(self):

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str) -> dict:
        r"""Load meta dict.
        """

        df = pd.read_csv(meta_csv, sep=',')

        indexes = df["split"].values == self.split

        meta_dict = {
            "midi_name": df["midi_filename"].values[indexes],
            "audio_name": df["audio_filename"].values[indexes],
            "duration": df["duration"].values[indexes]
        }

        return meta_dict



    def load_audio(self, path: str, start_time: float) -> dict:


        audio = load(
            path,
            sr=self.sr,
            mono= True,
            offset=start_time,
            duration=self.clip_duration
        )        

        audio = librosa.util.fix_length(data=audio, size=self.clip_samples, axis=-1)

        # shape: (channels, audio_samples)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": self.clip_duration
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_target(
        self, 
        midi_path: str, 
        start_time: float, 
        clip_duration: float
    ) -> dict:
        
        notes, pedals = read_single_track_midi(
            midi_path=midi_path, 
            extend_pedal=self.extend_pedal
        )

        data = {
            "note": notes,
            "pedal": pedals,
            "start_time": start_time,
            "clip_duration": clip_duration
        } 

        if self.target_transform:
            data = self.target_transform(data)

        return data

class Slakh2100(Dataset):
    def __init__(
        self, 
        root: str, 
        split: str = "train",
        sr: float = 16000,
        crop: Optional[callable] = RandomCrop(clip_duration=10., end_pad=9.9),
        transform: Optional[callable] = ToMono(),
        target: bool = True,
        extend_pedal: bool = True,
        target_transform: Optional[callable] = PianoRoll(fps=100, pitches_num=128),
    ):
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target = target
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform

        audios_dir = Path(self.root, self.split)
        self.meta_dict = {"audio_name": sorted(os.listdir(audios_dir))}

    def __getitem__(self, index: int) -> dict:
        prefix = Path(self.root, self.split, self.meta_dict["audio_name"][index])
        audio_path = Path(prefix, "mix.flac")
        meta_csv = Path(prefix, "metadata.yaml")
        midis_dir = Path(prefix, "MIDI")

        full_data = {
            "dataset_name": "Slakh2100",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load target (all combined tracks)
        if self.target:
            target_data = self.load_and_combine_tracks(
                meta_csv=meta_csv,
                midis_dir=midis_dir,
                start_time=audio_data["start_time"],
                clip_duration=audio_data["duration"]
            )
            full_data.update(target_data)

        return full_data

    def __len__(self):
        return len(self.meta_dict["audio_name"])

    def load_audio(self, path: str) -> dict:
        audio_duration = librosa.get_duration(path=path)
        
        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.
            clip_duration = None

        audio = load(
            path=path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration if clip_duration else audio_duration
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


def load_and_combine_tracks(self, meta_csv: str, midis_dir: str, start_time: float, clip_duration: float) -> dict:
    with open(meta_csv, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    combined_piano_roll = None  # To store the combined piano roll for all instruments

    for stem_name, stem_data in meta["stems"].items():
        if not stem_data["midi_saved"]:
            continue

        midi_path = Path(midis_dir, "{}.mid".format(stem_name))

        # Load individual MIDI tracks
        notes, pedals = read_single_track_midi(
            midi_path=midi_path, 
            extend_pedal=self.extend_pedal
        )

        # Prepare data dict with start_time and clip_duration
        data = {
            "note": notes,
            "pedal": pedals,
            "start_time": start_time,
            "clip_duration": clip_duration
        }

        # Convert to piano roll and combine by summing
        piano_roll = self.target_transform(data)

        if combined_piano_roll is None:
            combined_piano_roll = piano_roll
        else:
            combined_piano_roll += piano_roll  # Sum all tracks

    # Clip the combined piano roll to [0, 1] to ensure valid values
    combined_piano_roll = np.clip(combined_piano_roll, 0, 1)

    data = {
        "frame_roll": combined_piano_roll,  
        "start_time": start_time,
        "clip_duration": clip_duration
    }

    return data
