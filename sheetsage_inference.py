import os
import shutil
import numpy as np
from scipy.interpolate import interp1d
import torch
import torchaudio
import torchaudio.transforms as transforms
import pretty_midi as pyd
import demucs.separate
from data_processing.octMIDI import t2e, b2e, d2e, v2e, encoding_to_MIDI, encoding_to_str, CHECKTABLE
from utils.inference import read_leadsheet



def read_midi_by_sheetsage(midi_path, ACC=12):
    midi = pyd.PrettyMIDI(midi_path)
    beat_track = midi.instruments[0]
    beat_time = [note.start for note in beat_track.notes]
    downbeat_time = [note.start for note in beat_track.notes if note.pitch == 37]
    beat_time.sort()
    downbeat_time.sort()

    beat_metre = np.zeros(len(beat_time))
    for i, bt in enumerate(beat_time):
        if (bt in downbeat_time):
            beat_metre[i] = 1
    assert(sum(beat_metre)==len(downbeat_time))

    # get a beat table [1, 2, 3, 4, 1, 2, 3, ...], where '1' is the downbeat
    beats_table = np.zeros(len(beat_metre))
    for idx, value in enumerate(beat_metre):
        if value == 1:
            beats_table[idx] = 1
        elif value == 0:
            if beats_table[idx - 1] > 0:
                beats_table[idx] = beats_table[idx - 1] + 1
            else:
                beats_table[idx] = 0
    beats_table[beats_table > 8] = (beats_table[beats_table > 8] - 1) % 8 + 1
    beat_metre[beats_table==1] = 1
    # get a downbeat table [1, 1, 1, 1, 2, 2, 2, ...], where n denotes the n-th bar
    downbeats_table = np.cumsum(beat_metre)
    assert(len(beats_table) == len(downbeats_table))

    # get a time-signature table [4, 4, 4, 4, 6, 6, 6, ...], where n means n beats per bar
    ts_table = np.zeros(len(beats_table))
    last = 0
    highest = 0
    for idx, numerator in enumerate(beats_table[::-1]):
        if numerator >= last:
            ts_table[-1-idx] = numerator
            highest = numerator
        else:
            ts_table[-1-idx] = highest
        last = numerator
    assert((ts_table <= 8).all())

    # get a downbeat table [1, 1, 1, 1, 2, 2, 2, ...], where n denotes the n-th bar
    downbeats_table = np.cumsum(beat_metre)
    assert(len(beats_table) == len(downbeats_table))

    # get time table for each quantized 1/16 of quarter-beat
    beat_time = np.append(beat_time, beat_time[-1] + (beat_time[-1] - beat_time[-2]))
    quantize = interp1d(np.array(range(0, len(beat_time))) * ACC, beat_time, kind='linear')
    quavers = quantize(np.array(range(0, (len(beat_time) - 1) * ACC)))
    #print(quavers)
    quaver_table = np.zeros(len(beats_table) * ACC)
    for i in range(1, ACC+1):
        quaver_table[i-1::ACC] = (beats_table - 1) * ACC + i - 1
    downbeats_table = np.repeat(downbeats_table, ACC) - 1
    ts_table = np.repeat(ts_table, ACC, axis=0)
    assert(len(quavers) == len(downbeats_table))

    leadsheet_events = []
    for n_i, inst in enumerate(midi.instruments[1:]):
        for note in (inst.notes):
            if note.pitch < 6 or note.pitch >= 122:
                continue
            note_start = np.argmin(np.abs(quavers - note.start))
            note_end =  np.argmin(np.abs(quavers - note.end))
            dur = note_end - note_start
            bar = int(downbeats_table[note_start])
            if bar == -1:
                continue
            pos = int(quaver_table[note_start])
            ts = ts_table[note_start]

            current_time = note.start
            note_event = [current_time,\
                            bar, \
                            pos,\
                            4 if n_i==0 else 0,\
                            note.pitch if n_i==0 else note.pitch+12,\
                            d2e(dur),\
                            v2e(note.velocity),\
                            t2e((ts, 4)),\
                            b2e(120)
                        ]
            leadsheet_events.append(note_event)

    leadsheet_times = np.array([event[0] for event in leadsheet_events])
    leadsheet_events = np.array([event[1:] for event in leadsheet_events], dtype=int)
    seq_order = np.lexsort((-leadsheet_events[:, 3], leadsheet_events[:, 2], leadsheet_events[:, 1], leadsheet_events[:, 0]))

    leadsheet_times = leadsheet_times[seq_order]
    leadsheet_events = leadsheet_events[seq_order]

    return downbeat_time, leadsheet_events, leadsheet_times



def read_audio_to_codec(audio_path, audio_encoder, SR=32000, device='cuda:0', start=0, duration=20):
    waveform, sr = torchaudio.load(audio_path, channels_first=True)
    transform = transforms.Resample(sr, SR)
    waveform = transform(waveform)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform[:, int(start * SR): int((start + duration) * SR)]
    with torch.no_grad():
        waveform = waveform.to(device)
        codec = audio_encoder.encode(waveform.unsqueeze(0))
    return codec.audio_codes.squeeze(), waveform



def audio_voice_separate(source_path):
    demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", source_path])
    save_path = source_path.replace('original.wav', 'accompaniment.mp3')

    tmp_dir = f"separated/mdx_extra/original"

    shutil.move(os.path.join(tmp_dir, "no_vocals.mp3"), save_path)
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':

    from musecoco.hf_musecoco.midi_utils.utils_midi import RemiTokenizer
    from transformers import MusicgenForConditionalGeneration, MusicgenConfig
    from model_blip2 import Blip2Musecoco
    from dataset import collate_fn_inference, TK
    from tqdm import tqdm
    import traceback


    os.environ['CUDA_VISIBLE_DEVICES']= '0'
    DEVICE = 'cuda:0'

    # load audio encodec
    config=MusicgenConfig.from_pretrained("facebook/musicgen-small")
    config.decoder.return_dict_in_generate = True
    config.decoder.output_hidden_states = True
    auditory_encoder = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", config=config)
    audio_encoder = auditory_encoder.get_audio_encoder()
    audio_encoder.to(DEVICE)

    # load blip2musecoco model 
    model = Blip2Musecoco(load_pretrained=False)
    state_dict = torch.load("/data2/zhaojw/blip2_a2s/2025-08-12_202140_revamped_blip2_a2s_musecoco_PIAST_POP909_LoRA/models/revamped_blip2_a2s_musecoco_PIAST_POP909_LoRA_001_epoch.pt", map_location='cpu')
    #for key in list(state_dict.keys()):
    #    new_key = key.replace('blip2qformer.ln_audio', 'blip2qformer.layer_norm')
    #    state_dict[new_key] = state_dict.pop(key)  
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    audio_root_dir = '/data2/zhaojw/a2s_style_learning/crazy_piano'
    #midi_root_dir = '/data2/zhaojw/a2s_style_learning/hooktheory/midi'
    midi_root_dir = '/data2/zhaojw/a2s_style_learning/pop1k7_ls'

    START_TIME = {
        'alkan_op_17': 275,
        'bartok_sonata_80': 3,
        'chopin_ocean': 5,
        'chopin_revolution': 5,
        'chopin_winter': 25,
        'Kapustin_etude_3': 6,
        'Liszt_feux_follets': 80,
        'Liszt_hungarian_2': 462,
        'prokofiev_concerto_2': 492,
        'rachmaninoff_etude_6': 2,
        'Scriabin_sonata_5': 625,
        'sziffra_bumblebee': 3
    }


    SAVE_ROOT = '/data2/zhaojw/a2s_style_learning/crazy_piano_cover_new_model_k=5'
    for midi in os.listdir(midi_root_dir)[:12]:
        #midi = '74.mid'
        #print(midi)
        for audio in os.listdir(audio_root_dir):
            #audio = 'Liszt_hungarian_2.mp3'
            #print(audio)
            audio_path = os.path.join(audio_root_dir, audio)
            #process audio from waveform to codec
            codec, waveform = read_audio_to_codec(audio_path, audio_encoder, device=DEVICE, start=START_TIME[audio.replace('.mp3', '')])
            codec = codec.detach().cpu().numpy()
            waveform = waveform.detach().cpu()

            save_path = os.path.join(SAVE_ROOT, audio.replace('.mp3', ''))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                torchaudio.save(os.path.join(save_path, audio), waveform, 32000)
            

            midi_path = os.path.join(midi_root_dir, midi)

            #downbeat_time, leadsheet_events, leadsheet_times = read_midi_by_sheetsage(midi_path)
            #downbeat_time.append(downbeat_time[-1] + (downbeat_time[-1] - downbeat_time[-2]))
            downbeat_time, leadsheet_events, leadsheet_times = read_leadsheet(midi_path)
            #print(leadsheet_events)
            #downbeat_time.append(downbeat_time[-1] + (downbeat_time[-1] - downbeat_time[-2]))
            #downbeat_time.append(downbeat_time[-1] + (downbeat_time[-1] - downbeat_time[-2]))

            # process leadsheet into REMI format
            HOP_LEN=2
            BAR_LEN = 4
            STAR_BAR = 0
            DURATION = leadsheet_events[-1][0]-STAR_BAR#len(downbeat_time)
            audio_slices, audio_mask, ls_slices = collate_fn_inference([codec, downbeat_time, None, None, leadsheet_times, leadsheet_events], DEVICE, [STAR_BAR, STAR_BAR+DURATION], HOP_LEN, BAR_LEN, True)
            #print(audio_slices.shape, audio_mask.shape, [ls_slices[idx].shape for idx in range(len(ls_slices))])

            #generate with musecoco
            generation_slices = []
            intervals = []
            for idx in range(len(audio_slices)):
                #print('slice', idx, 'of', len(audio_slices))
                decoder_input_ids = ls_slices[idx]
                #print('decoder_input_ids', decoder_input_ids.shape)
                #print(decoder_input_ids[0])
                if idx > 0:
                    enter_point = torch.nonzero(generation_slices[-1]==5)[HOP_LEN-1, 1] + 1
                    try:
                        break_point = torch.nonzero(generation_slices[-1]==5)[BAR_LEN-1, 1] + 1
                    except IndexError:
                        break_point = -1
                    #print('generation_slices', len(generation_slices), generation_slices[-1].shape)
                    decoder_input_ids = torch.cat([decoder_input_ids, generation_slices[-1][:, enter_point:]], dim=1)
                    intervals.append((enter_point, None))
                
                token_pred = model.generate_nucleus2(audio_slices[idx: idx+1], decoder_input_ids, audio_mask[idx: idx+1], t=1, p=None, k=5)
                #print('token_pred', token_pred.shape)
                #print(token_pred)

                enter_point = torch.nonzero(token_pred==984)[-1, 1] + 1
                token_pred = token_pred[:, enter_point:]

                generation_slices.append(token_pred)

            generation_reult = torch.cat([generation_slices[idx][:, :itv[0]] \
                                    for idx, itv in enumerate(intervals)] \
                                        + [generation_slices[-1]], 
                                        dim=1)
            pred_sample = [TK._convert_id_to_token(tk) for tk in generation_reult[0].detach().cpu().numpy()]
            #print('pred_sample', pred_sample)
            pred_sample = [tk for tk in pred_sample if ('-' in tk)]
            midi_tok = RemiTokenizer()
            pred_sample = midi_tok.remi_to_midi(pred_sample, ignore_velocity=False)

            save_name = os.path.join(save_path, midi)
            pred_sample.dump(save_name)

            # add pedal to the piano cover
            start_bar = leadsheet_events[(leadsheet_times >= downbeat_time[STAR_BAR])][0, 0]
            leadsheet_events = leadsheet_events[np.array(leadsheet_events[:, 0] >= start_bar) & np.array(leadsheet_events[:, 0] < start_bar+DURATION)]
            leadsheet_events[:, 0] -= start_bar
            leadsheet_events = leadsheet_events[leadsheet_events[:, 2]==4]  #chord
            chord_change_steps = list(range(0, DURATION*4, 4))
            beats_per_bar = int(np.ceil(max(leadsheet_events[:, 1]+leadsheet_events[:, 4])/12))
            for i in range(len(leadsheet_events)):
                step = leadsheet_events[i, 0] * beats_per_bar + leadsheet_events[i, 1] // 12
                if not step in chord_change_steps:
                    chord_change_steps.append(step)
            chord_change_steps.sort()
            #print(chord_change_steps)

            pred_sample = pyd.PrettyMIDI(save_name)
            beat_times = pred_sample.get_beats()
            pedals = []
            for i in range(len(chord_change_steps)-1):
                try:
                    pedals.append(pyd.ControlChange(number=64, value=72, time=beat_times[chord_change_steps[i]]))
                    pedals.append(pyd.ControlChange(number=64, value=0, time=beat_times[chord_change_steps[i+1]]-0.05))
                except IndexError:
                    break
            pred_sample.instruments[0].control_changes = pedals
            pred_sample.write(save_name)
            #import sys
            #sys.exit()
