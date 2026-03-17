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
output_hdf5_path = "/data1/zhaojw/PIAST/audio_data.h5"


config=MusicgenConfig.from_pretrained("facebook/musicgen-small")
config.decoder.return_dict_in_generate = True
config.decoder.output_hidden_states = True
#model = MusicgenForConditionalGeneration(config=config)
auditory_encoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", config=config)
auditory_encoder.to('cuda:0')
audio_encoder = auditory_encoder.get_audio_encoder()
audio_encoder.to('cuda:0')



SR = 32000

#count = 0
#total = 10
with h5py.File(output_hdf5_path, 'w') as hdf5_file:
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
        if time_shift < 0:
            waveform = torch.nn.functional.pad(waveform, (-int(time_shift*SR), 0), 'constant', 0)
        elif time_shift > 0:
            waveform = waveform[:, int(time_shift*SR):]
        
        try:
            with torch.no_grad():
                waveform = waveform.to('cuda:0')
                codec = audio_encoder.encode(waveform.unsqueeze(0))
                #codec.append(codec.audio_codes[0,  :, ...])#channel, batch, codebook, seq_len = codec.audio_codes.shape
                #print(codec.audio_codes.shape)
        except torch.OutOfMemoryError:
            print(f"Out of memory for {audio_file}")
            continue

        dataset_name = audio_file.replace('.mp3', '')
        hdf5_file.create_dataset(dataset_name, data=codec.audio_codes.squeeze().detach().cpu().numpy())
        
        #count += 1
        #if count == total:
        #    break

"""with h5py.File(output_hdf5_path, 'r') as hdf5_file:
    for dataset_name in hdf5_file.keys():
        data = hdf5_file[dataset_name]
        print(type(data))
        print(f"Dataset {dataset_name} has shape {data.shape} and dtype {data.dtype}")"""
