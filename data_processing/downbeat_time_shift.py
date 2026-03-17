import os
import librosa
import pretty_midi as pyd
import numpy as np
import h5py
from tqdm import tqdm

import torchaudio
import torchaudio.transforms as transforms
import torch

from transformers import MusicgenForConditionalGeneration, MusicgenConfig


audio_dir = "/data1/zhaojw/PIAST/PIAST/piast_yt/audio/"
midi_dir = "/data1/zhaojw/PIAST/PIAST/piast_yt/midi/"
downbeat_dir = "/data1/zhaojw/PIAST/PIAST/piast_yt/audio_downbveat_tracking/"
downbeat_write_dir = "/data1/zhaojw/PIAST/PIAST/piast_yt/audio_downbeat_align_midi/"




SR = 32000

#count = 0
#total = 10

for audio_file in tqdm(os.listdir(audio_dir)):
    audio_path = os.path.join(audio_dir, audio_file)
    #y, sr = librosa.load(audio_path, mono=True)
    waveform, sr = torchaudio.load(audio_path, channels_first=True)
    transform = transforms.Resample(sr, SR)
    waveform = transform(waveform)
    waveform = torch.mean(waveform, dim=0, keepdim=True)

    midi = pyd.PrettyMIDI(os.path.join(midi_dir, audio_file.replace('.mp3', '.mid')))
    inst = midi.instruments[0]    #discard melody track
    midi_start = sorted([note.start for note in inst.notes])[0]
    audio_start = sorted(librosa.onset.onset_detect(y=waveform[0].numpy(), sr=SR, units='time'))[0]
    time_shift = audio_start - midi_start
    #print('time_shift', time_shift)

    beat_annotation = np.loadtxt(os.path.join(downbeat_dir, audio_file.replace('.mp3', '.txt')))
    #print(beat_annotation[:10])
    beat_annotation[:, 0] -= time_shift
    beat_annotation = beat_annotation[beat_annotation[:, 0] >= 0]
    #print(beat_annotation[:10])

    np.savetxt(os.path.join(downbeat_write_dir, audio_file.replace('.mp3', '.txt')), beat_annotation)




