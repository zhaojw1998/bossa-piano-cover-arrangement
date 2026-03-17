import os
from transformers import MusicgenForConditionalGeneration, MusicgenConfig
import h5py
from tqdm import tqdm
import torchaudio
import torch
import torchaudio.transforms as transforms

audio_dir = "/data1/zhaojw/Q&A/POP909-Dataset/pop909_original&accompaniment/"
output_hdf5_path = "/data1/zhaojw/Q&A/POP909-Dataset/pop909_audio.h5"


config=MusicgenConfig.from_pretrained("facebook/musicgen-small")
config.decoder.return_dict_in_generate = True
config.decoder.output_hidden_states = True
#model = MusicgenForConditionalGeneration(config=config)
auditory_encoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", config=config)
auditory_encoder.to('cuda:0')
audio_encoder = auditory_encoder.get_audio_encoder()
audio_encoder.to('cuda:0')



SR = 32000

with h5py.File(output_hdf5_path, 'w') as hdf5_file:
    for audio_name in tqdm(os.listdir(audio_dir)):
        audio_path = os.path.join(audio_dir, audio_name, 'accompaniment.mp3')
        #y, sr = librosa.load(audio_path, mono=True)
        waveform, sr = torchaudio.load(audio_path, channels_first=True)
        transform = transforms.Resample(sr, SR)
        waveform = transform(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        try:
            with torch.no_grad():
                waveform = waveform.to('cuda:0')
                codec = audio_encoder.encode(waveform.unsqueeze(0))
        except torch.OutOfMemoryError:
            print(f"Out of memory for {audio_name}")
            continue
        hdf5_file.create_dataset(audio_name, data=codec.audio_codes.squeeze().detach().cpu().numpy())
        