import os
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.octMIDI import e2t, e2d, e2v, e2b, note_encode
from utils.REMI import note2token, get_musecoco_attributes




# Constants for dataset parameters
BAR_LEN = 4         # 4-bar MIDI samples
HOP_LEN = 2         # 2-bar MIDI hop length
NOTE_LEN = 224      # max notes per MIDI sample
ACC = 12            # MIDI quantization resolution at 1/12 per beat

AUD_LEN = 10.16     # 10s audio samples
SR = 50             # audio sample rate by MusicGEN

# Directories for datasets
SPLIT_DICT_DIR = '/data2/zhaojw/audio_midi_data/dataset_split.json' # specifies train/val/test split for PIAST and POP909 datasets
# Audio codec
PIAST_CODEC_DIR = "/data2/zhaojw/audio_midi_data/PIAST_codec.h5"
POP909_CODEC_DIR = "/data2/zhaojw/audio_midi_data/POP909_codec.h5"
# Symbolic MIDI (processed as note even sequences)
PIAST_MIDI_DIR = "/data2/zhaojw/audio_midi_data/PIAST_octMIDI_12bins/"
POP909_MIDI_DIR = "/data2/zhaojw/audio_midi_data/pop909_octMIDI_12bins/"



class Ensemble_Dataset(Dataset):
    """Ensemble dataset combining PIAST and POP909 datasets for training and evaluation."""
    def __init__(self, debug_mode=False, split='train', load_num=None, pitch_order='low_to_high'):
        super(Ensemble_Dataset, self).__init__()
        # debug_mode: if True, only load a small subset of the dataset for debugging
        # split: 'train', 'validation', or 'test'
        # load_num: number of samples to load for debugging, if debug_mode is True
        # pitch_order: 'low_to_high' or 'high_to_low', determines the order of simutaneous notes
        
        with open(SPLIT_DICT_DIR, 'r') as f:
            split_dict = json.load(f)
        
        print('loading PIAST dataset...')
        self.piast = AudioSymbolicDataset(codec_dir=PIAST_CODEC_DIR,
                                        midi_dir=PIAST_MIDI_DIR,
                                        bar_len=BAR_LEN, hop_len=HOP_LEN, audio_len=AUD_LEN, sampling_rate=SR,
                                        split=split,
                                        debug_mode=debug_mode,
                                        load_num=load_num,
                                        pitch_order=pitch_order,
                                        split_dict=split_dict['PIAST']
                                        )
        
        print('loading POP909 dataset...')
        self.pop909 = AudioSymbolicDataset(codec_dir=POP909_CODEC_DIR,
                                        midi_dir=POP909_MIDI_DIR,
                                        bar_len=BAR_LEN, hop_len=HOP_LEN, audio_len=AUD_LEN, sampling_rate=SR,
                                        split=split,
                                        debug_mode=debug_mode,
                                        load_num=load_num,
                                        pitch_order=pitch_order,
                                        split_dict=split_dict['POP909']
                                        )
        
        self.lens = [len(self.piast), len(self.pop909)]

    def __len__(self):
        return sum(self.lens)
    
    def __getitem__(self, idx):
        if idx < self.lens[0]:
            return self.piast[idx]
        else:
            return self.pop909[idx-self.lens[0]]



class AudioSymbolicDataset(Dataset):
    def __init__(self, codec_dir, midi_dir, bar_len=BAR_LEN, hop_len=HOP_LEN, audio_len=AUD_LEN, sampling_rate=SR, debug_mode=False, split='train', load_num=None, pitch_order='low_to_high', split_dict=None):
        super(AudioSymbolicDataset, self).__init__()

        self.codec_dir = codec_dir
        self.midi_dir = midi_dir
        self.bar_len = bar_len
        self.hop_len = hop_len
        self.debug_mode = debug_mode
        self.split = split
        self.audio_len = audio_len
        self.sr = sampling_rate
        self.load_num = load_num
        self.pitch_order = pitch_order

        self.split_dict = split_dict
        if self.split_dict is not None:
            self.validation_set = self.split_dict['validation']
            self.test_set = self.split_dict['test']

        self.memory = dict({'audio': [],
                            'note_times': [],
                            'note_events': [],
                            'leadsheet_times': [],
                            'leadsheet_events': [],
                            'indices': []
                            })
        self.load_data()

    def __len__(self):
        return len(self.memory['indices'])
    
    def __getitem__(self, idx):
        sample = self.memory['indices'][idx]
        
        song_idx = sample['song_idx']
        start_time, end_time = sample['time_range']

        # load arrangement segment
        note_times = self.memory['note_times'][song_idx]
        note_events = self.memory['note_events'][song_idx]

        tmp = note_events[note_times >= start_time]
        start_bar = tmp[0, 0]
        note_sample = note_events[(note_events[:, 0] >= start_bar) & (note_events[:, 0] < start_bar + self.bar_len)]

        # load lead sheet segment
        #leadsheet_times = self.memory['leadsheet_times'][song_idx]
        leadsheet_events = self.memory['leadsheet_events'][song_idx]

        #tmp = leadsheet_events[leadsheet_times >= start_time]
        #start_bar = tmp[0, 0]
        leadsheet_sample = leadsheet_events[(leadsheet_events[:, 0] >= start_bar) & (leadsheet_events[:, 0] < start_bar + self.bar_len)]

        #load audio segment
        audio = self.memory['audio'][song_idx]
        #start_frame = start_time * self.sr
        #end_frame = end_time * self.sr
        mid_frame = int(round((start_time + end_time) * self.sr // 2))
        audio_sample_len = int(self.audio_len * self.sr)
        start_frame = mid_frame - audio_sample_len // 2
        end_frame = mid_frame + audio_sample_len // 2
        #start_frame = np.random.randint(max(start_frame-audio_sample_len, 0), min(end_frame, audio.shape[1]-audio_sample_len))
        audio_sample = audio[:, start_frame: min(end_frame, audio.shape[1])]

        return audio_sample, note_sample, leadsheet_sample
        

    def load_data(self): 
        with h5py.File(self.codec_dir, "r") as f:
            all_songs =list(f.keys())

            if self.split_dict is not None:
                if self.split == 'validation':
                    data_to_load = self.split_dict['validation']
                elif self.split == 'test':
                    data_to_load = self.split_dict['test']
                elif self.split == 'train':
                    data_to_load = [song for song in all_songs if ((song not in self.split_dict['validation']) and (song not in self.split_dict['test']))]
            else:
                if self.split == 'train':
                    data_to_load = all_songs[:int(len(all_songs) * 0.9)]
                elif self.split == 'validation':
                    data_to_load = all_songs[int(len(all_songs) * 0.9): int(len(all_songs) * 0.95)]
                elif self.split == 'test':
                    data_to_load = all_songs[int(len(all_songs) * 0.95):]
            if self.debug_mode:
                data_to_load = data_to_load[: self.load_num] if self.load_num is not None else data_to_load[:10]
        
            for song_name in tqdm(data_to_load):
                midi_data = np.load(os.path.join(self.midi_dir, song_name+'.npz'))
                
                downbeat_times = midi_data['downbeat_times']
                note_times = midi_data['note_times']
                note_events = midi_data['note_events']
                leadsheet_times = midi_data['leadsheet_times']
                leadsheet_events = midi_data['leadsheet_events']
                max_time = max(note_times)

                if (note_events[:, 1] > 127).any() or (leadsheet_events[:, 1] > 127).any():
                    print('position out range error, skip')
                    continue      

                for idx in range(0, len(downbeat_times)-self.bar_len, self.hop_len):
                    
                    if int(downbeat_times[idx+self.bar_len] * self.sr) > f[song_name].shape[1]:
                        break

                    if downbeat_times[idx+self.bar_len] > max_time:
                        break

                    tmp = note_events[note_times >= downbeat_times[idx]]
                    start_bar = tmp[0, 0]

                    note_sample = note_events[(note_events[:, 0] >= start_bar) & (note_events[:, 0] < start_bar + self.bar_len)]
                    leadsheet_sample = leadsheet_events[(leadsheet_events[:, 0] >= start_bar) & (leadsheet_events[:, 0] < start_bar + self.bar_len)]

                    if len(note_sample) == 0 or len(leadsheet_sample) == 0:
                        continue

                    self.memory['indices'].append({'song_idx': len(self.memory['audio']),
                                                'time_range': (downbeat_times[idx], downbeat_times[idx+self.bar_len]), #downbeat_times[idx+n_bars-1]
                                                })
                    
                self.memory['audio'].append(f[song_name][()])
                if self.pitch_order == 'low_to_high':
                    note_order = np.lexsort((note_events[:, 3], note_events[:, 2], note_events[:, 1], note_events[:, 0]))
                    leadsheet_order = np.lexsort((leadsheet_events[:, 3], leadsheet_events[:, 2], leadsheet_events[:, 1], leadsheet_events[:, 0]))
                elif self.pitch_order == 'high_to_low':
                    note_order = np.lexsort((-note_events[:, 3], note_events[:, 2], note_events[:, 1], note_events[:, 0]))
                    leadsheet_order = np.lexsort((-leadsheet_events[:, 3], leadsheet_events[:, 2], leadsheet_events[:, 1], leadsheet_events[:, 0]))
                
                self.memory['note_times'].append(note_times[note_order])
                self.memory['note_events'].append(note_events[note_order])
                self.memory['leadsheet_times'].append(leadsheet_times[leadsheet_order])
                self.memory['leadsheet_events'].append(leadsheet_events[leadsheet_order])



def collate_fn_musicbert(batch, device, max_time=int(AUD_LEN*SR), max_note=NOTE_LEN+2, augment=True, deperf=False):
    """collate_fn for Stage 1 training, where music notes are packed in MusicBERT's OctMIDI format.
    """
    # deperf: if True, discard velocity and tempo information in lead sheet

    audios = []         #codec
    notes = []          #note token
    audio_masks = []    #codec pad mask
    ls_masks = []       #token lead sheet mask
    pad_masks = []      #token pad mask

    for audio, track, leadsheet in batch:
        audio_mask = np.ones(max_time + 4)  # 4 results from delayed interleave
        audio_pad_len = max_time - audio.shape[1]
        if audio_pad_len > 0:
            audio = np.pad(audio, ((0, 0), (0, audio_pad_len)), 'constant', constant_values=2048)
            audio_mask[-audio_pad_len:] = 0
        if audio.shape[1] > max_time:
            audio = audio[:, :max_time]

        if augment:
            #pitch augmentation
            pitch_shift = np.random.randint(-6, 6)    
            track[track[:, 2] != 128, 3] += pitch_shift
            leadsheet[:, 3] += pitch_shift
            #tempo augmentation
            #bpm_shift = 0.5 + np.random.rand()  # 0.5 ~ 1.5
            #track[:, 7] = b2e(bpm_shift * e2b(track[:, 7]))

        assert(len(leadsheet) > 0 and len(track) > 0)
        offset = min(leadsheet[0, 0], track[0, 0])

        leadsheet = note_encode(leadsheet, bar_index_offset=-offset)[8: -8]
        if deperf:
            leadsheet[5::8] = 3 #discard velocity
            leadsheet[7::8] = 3 #discard tempo

        track = note_encode(track, bar_index_offset=-offset)
        note = np.concatenate([leadsheet, track], axis=0)
        note = note[:min(len(note)*8, max_note*8)]

        ls_mask = np.ones(max_note)
        ls_mask[:len(leadsheet)//8] = 0

        pad_mask = np.ones(max_note)
        if (len(note) // 8) < max_note:
            pad_len = max_note - (len(note) // 8)
            note = np.pad(note, (0, 8*pad_len), 'constant', constant_values=1)
            pad_mask[-pad_len:] = 0

       
        audios.append(audio)
        notes.append(note)
        audio_masks.append(audio_mask)
        ls_masks.append(ls_mask)
        pad_masks.append(pad_mask)

    return torch.from_numpy(np.array(audios)).to(device), \
        torch.from_numpy(np.array(notes)).to(device).contiguous(), \
        torch.from_numpy(np.array(audio_masks)).to(device).contiguous(), \
        torch.from_numpy(np.array(ls_masks)).to(device), \
        torch.from_numpy(np.array(pad_masks)).to(device).contiguous()




from musecoco.hf_musecoco.tokenization_musecoco import MuseCocoTokenizer
from musecoco.hf_musecoco.midi_utils.midiprocessor.vocab_manager import VocabManager

MODEL_SIZE = '1b'
PRETRAINED_MUSECOCO_PATH = '/data2/zhaojw/LAVIS/musecoco'
VM = VocabManager()
TK = MuseCocoTokenizer.from_pretrained(os.path.join(PRETRAINED_MUSECOCO_PATH, MODEL_SIZE, 'tokenizer'))



def collate_fn_musecoco2(batch, device, max_time=int(AUD_LEN*SR), max_token=1000, augment=True, deperf=False):
    """collate_fn for Stage 2 training, where music tokens are converted to MuseCoco's REMI format.
    """
    # deperf: if True, discard velocity and tempo information in lead sheet

    audios = []
    tokens = []
    audio_masks = []
    control_pad_masks = []      # masks attribute control tokens
    token_pad_masks = []        # masks pad tokens

    for audio, track, leadsheet in batch:

        if augment: #pitch augmentation
            pitch_shift = np.random.randint(-6, 6)    
            track[track[:, 2] != 128, 3] += pitch_shift
            leadsheet[:, 3] += pitch_shift

        #convert octMIDI to musecoco's REMI format
        track[track[:, 2]==128, 3] = np.array([VM.convert_pitch_to_id(pitch-128, True) for pitch in track[track[:, 2]==128, 3]])
        track[track[:, 2]!=128, 3] = np.array([VM.convert_pitch_to_id(pitch, False) for pitch in track[track[:, 2]!=128, 3]])
        track[:, 4] = np.array([VM.convert_dur_to_id(e2d(dur)) for dur in track[:, 4]])
        track[:, 5] = np.array([VM.convert_vel_to_id(e2v(tempo)) for tempo in track[:, 5]])
        track[:, 6] = np.array([VM.convert_ts_to_id(e2t(ts)) for ts in track[:, 6]])
        track[:, 7] = np.array([VM.convert_tempo_to_id(e2b(bpm)) for bpm in track[:, 7]])

        leadsheet[:, 3] = np.array([VM.convert_pitch_to_id(pitch, False) for pitch in leadsheet[:, 3]])
        leadsheet[:, 4] = np.array([VM.convert_dur_to_id(e2d(dur)) for dur in leadsheet[:, 4]])
        leadsheet[:, 5] = np.array([VM.convert_vel_to_id(e2v(tempo)) for tempo in leadsheet[:, 5]])
        leadsheet[:, 6] = np.array([VM.convert_ts_to_id(e2t(ts)) for ts in leadsheet[:, 6]])
        leadsheet[:, 7] = np.array([VM.convert_tempo_to_id(e2b(bpm)) for bpm in leadsheet[:, 7]])

        attribute_tokens = get_musecoco_attributes()  #piano only
        try:
            offset = min(track[0, 0], leadsheet[0, 0])
        except IndexError:
            print("Error: Empty track or leadsheet, skipping this sample.")
            print("Track:", track.shape, track)
            print("Leadsheet:", leadsheet.shape, leadsheet)
            continue
        if offset > 0:   #delete empty bars in the beginning if any
            track[:, 0] -= offset
            leadsheet[:, 0] -= offset

        track_token = note2token(track)
        leasheet_token = note2token(leadsheet, redact_performance=deperf)
        token_list = ['</s>'] + attribute_tokens + leasheet_token + ['<sep>'] + track_token + ['</s>']
        token_list = token_list[:min(len(token_list), max_token)]

        tk_pad_len = max_token - len(token_list)
        token_mask = np.zeros(max_token)  # 2 result from sos and eos
        if tk_pad_len > 0:
            token_list = token_list + ['<pad>'] * tk_pad_len
            token_mask[-tk_pad_len:] = 1    #for musecoco, 1 means masked positions

        cntl_mask = np.zeros(max_token)
        cntl_mask[:-(len(track_token)+1+tk_pad_len)] = 1

        # pack and pad audio codec
        audio_mask = np.ones(max_time + 4)  # 4 results from delayed interleave
        audio_pad_len = max_time - audio.shape[1]
        if audio_pad_len > 0:
            audio = np.pad(audio, ((0, 0), (0, audio_pad_len)), 'constant', constant_values=2048)
            audio_mask[-audio_pad_len:] = 0
        if audio.shape[1] > max_time:
            audio = audio[:, :max_time]
        
        audios.append(audio)
        tokens.append(' '.join(token_list))
        audio_masks.append(audio_mask)
        control_pad_masks.append(cntl_mask)
        token_pad_masks.append(token_mask)      

    return torch.from_numpy(np.array(audios)).to(device).contiguous(), \
            TK(tokens, return_tensors="pt")['input_ids'][:, :-1].to(device), \
            torch.from_numpy(np.array(audio_masks)).to(device).contiguous(), \
            torch.BoolTensor(np.array(control_pad_masks)).to(device).contiguous(), \
            torch.BoolTensor(np.array(token_pad_masks)).to(device).contiguous()



    
def collate_fn_inference(batch, device, db_range=(0, -1), hop_len=2, bar_len=4, deperf=True):
    """collate_fn at inference time, packing a long piece into batches. Input batch size should be 1."""
    # deperf: if True, discard velocity and tempo information in lead sheet
    
    audio_slices = []
    audio_masks = []
    leadsheet_slices = []

    audio, downbeat, _, _, leadsheet_t, leadsheet_e = batch if len(batch) > 1 else batch[0]

    leadsheet_e[:, 3] = np.array([VM.convert_pitch_to_id(pitch, False) for pitch in leadsheet_e[:, 3]])
    leadsheet_e[:, 4] = np.array([VM.convert_dur_to_id(e2d(dur)) for dur in leadsheet_e[:, 4]])
    leadsheet_e[:, 5] = np.array([VM.convert_vel_to_id(e2v(tempo)) for tempo in leadsheet_e[:, 5]])
    leadsheet_e[:, 6] = np.array([VM.convert_ts_to_id(e2t(ts)) for ts in leadsheet_e[:, 6]])
    leadsheet_e[:, 7] = np.array([VM.convert_tempo_to_id(e2b(bpm)) for bpm in leadsheet_e[:, 7]])

    downbeat = downbeat[db_range[0]: min(db_range[1]+2, len(downbeat))]
    for idx in range(0, len(downbeat)-bar_len, hop_len):
        if idx+bar_len >= len(downbeat):
            break

        if idx == 0:
            tmp = leadsheet_e[(leadsheet_t >= downbeat[idx]) & (leadsheet_t < downbeat[idx+bar_len])]
            start_bar = tmp[0, 0]
        else:
            start_bar += hop_len

        leadsheet_slice = leadsheet_e[np.array(leadsheet_e[:, 0] >= start_bar) & np.array(leadsheet_e[:, 0] < start_bar+bar_len)]
        leadsheet_slice = note2token(leadsheet_slice, redact_performance=deperf)

        attribute_tokens = get_musecoco_attributes()  #piano only
        leadsheet_slice = TK(' '.join(['</s>'] + attribute_tokens + leadsheet_slice + ['<sep>']), return_tensors="pt")['input_ids'][:, :-1].to(device)
        
        leadsheet_slices.append(leadsheet_slice)

        mid_frame = int(round((downbeat[idx] + downbeat[idx+bar_len]) * SR // 2))
        audio_sample_len = int(AUD_LEN * SR)
        start_frame = mid_frame - audio_sample_len // 2
        end_frame = mid_frame + audio_sample_len // 2
        if start_frame < 0:
            start_frame = 0
            end_frame = audio_sample_len
        elif end_frame > audio.shape[1]:
            end_frame = audio.shape[1]
            start_frame = end_frame - audio_sample_len
        audio_slice = audio[:, start_frame: end_frame]

        audio_mask = np.ones(audio_sample_len + 4)  # 4 results from delayed interleave
        audio_pad_len = audio_sample_len - audio_slice.shape[1]
        if audio_pad_len > 0:
            audio_slice = np.pad(audio_slice, ((0, 0), (0, audio_pad_len)), 'constant', constant_values=2048)
            audio_mask[-audio_pad_len:] = 0
        if audio_slice.shape[1] > audio_sample_len:
            audio_slice = audio_slice[:, :audio_sample_len]

        audio_slices.append(audio_slice)
        audio_masks.append(audio_mask)

    return torch.from_numpy(np.array(audio_slices)).to(device), \
            torch.from_numpy(np.array(audio_masks)).to(device).contiguous(), \
            leadsheet_slices


                    
if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    from transformers import MusicgenForConditionalGeneration, MusicgenConfig
    from utils.octMIDI import str_to_encoding, encoding_to_MIDI, CHECKTABLE
    import soundfile as sf
    from musecoco.hf_musecoco.midi_utils.utils_midi import RemiTokenizer


    os.environ['CUDA_VISIBLE_DEVICES']= '6'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    config=MusicgenConfig.from_pretrained("facebook/musicgen-small")
    auditory_encoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", config=config)
    audio_encoder = auditory_encoder.get_audio_encoder()
    audio_encoder.to('cuda:0')

    
    """dataset = PIAST_Dataset(codec_dir="/data1/zhaojw/Q&A/POP909-Dataset/pop909_audio.h5",
                            midi_dir="/data1/zhaojw/Q&A/POP909-Dataset/pop909_midi_12bins/",
                            split='train',
                            debug_mode=True,
                            load_num=10,
                            pitch_order='high_to_low',
                            )"""
    
    dataset = Ensemble_Dataset(debug_mode=True, split='train', load_num=10, pitch_order='high_to_low')

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: collate_fn_musecoco2(x, 'cuda:0', augment=True))
    for audio, notes, audio_mask, leadsheet_mask, pad_mask in dataloader:
        
        torch.set_printoptions(profile="full")
        #print(leadsheet[0])
        #print(notes[0])
        print(audio.shape, notes.shape, audio_mask.shape, leadsheet_mask.shape, pad_mask.shape)
        #print('leadsheet pad', torch.sum(leadsheet==1, dim=-1))
        print('note pad', (torch.sum(notes==1, dim=-1)) / notes.shape[-1])


        audio[audio == 2048] = 0
        output_values = audio_encoder.decode(
            audio.unsqueeze(0),
            audio_scales=[None]*len(audio),
        )
        #print('output_values', output_values.audio_values.shape)


        audio_save_dir = 'dataload_test_audio'
        midi_save_dir = 'dataload_test_midi'
        if not os.path.exists(audio_save_dir):
            os.makedirs(audio_save_dir)
        if not os.path.exists(midi_save_dir):
            os.makedirs(midi_save_dir)
        reverse_checkltable = {v: k for k, v in CHECKTABLE.items()}
        for bs in range(len(notes)):
            sf.write(os.path.join(audio_save_dir, f'{str(bs).zfill(2)}.wav'), output_values.audio_values[bs][0].detach().cpu().numpy(), 32000, 'PCM_24')

            track = notes[bs][leadsheet_mask[bs].bool()].detach().cpu().numpy()

            pred_sample = [TK._convert_id_to_token(tk) for tk in track]
            pred_sample = [tk for tk in pred_sample if ('-' in tk)]
            midi_tok = RemiTokenizer()
            midi = midi_tok.remi_to_midi(pred_sample, ignore_velocity=False)
            #pred_sample.dump(os.path.join(save_path, f'{str(num).zfill(2)}.mid'))
            print('----------', bs, '----------')
            print(pred_sample)


            track = notes[bs][~leadsheet_mask[bs].bool()].detach().cpu().numpy()
            
            pred_sample = [TK._convert_id_to_token(tk) for tk in track]
            pred_sample = [tk for tk in pred_sample if ('-' in tk)]
            midi_tok = RemiTokenizer()
            pred_sample = midi_tok.remi_to_midi(pred_sample, ignore_velocity=False)

            midi.instruments += pred_sample.instruments
            midi.dump(os.path.join(midi_save_dir, f'{str(bs).zfill(2)}.mid'))
        break

