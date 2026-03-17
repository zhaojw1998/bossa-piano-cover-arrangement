import torch
import pretty_midi as pyd
import numpy as np
from scipy.interpolate import interp1d
from .octMIDI import t2e, b2e, d2e, v2e, encoding_to_MIDI, encoding_to_str, CHECKTABLE
import torchaudio
import torchaudio.transforms as transforms
import os

def nucleus_filter(logits, p):
    #sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    #cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    cum_sum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    #sorted_indices_to_remove = cumulative_probs > p
    nucleus = cum_sum_probs < p
    # Shift the indices to the right to keep also the first token above the threshold
    #sorted_indices_to_remove = torch.cat([sorted_indices_to_remove.new_zeros(sorted_indices_to_remove.shape[:-1] + (1,)), sorted_indices_to_remove[..., :-1]], dim=-1)
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    nucleus = nucleus.gather(-1, sorted_indices.argsort(-1))

    logits[~nucleus] = float('-inf')
    return logits



def read_audio_to_codec(audio_path, audio_encoder, SR=32000, device='cuda:0', offset=0):
    waveform, sr = torchaudio.load(audio_path, channels_first=True)
    # Trim the waveform to start from the 10th second
    start_sample = int(offset * sr)
    waveform = waveform[:, start_sample:]
    transform = transforms.Resample(sr, SR)
    waveform = transform(waveform)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    with torch.no_grad():
        waveform = waveform.to(device)
        codec = audio_encoder.encode(waveform.unsqueeze(0))
    return codec.audio_codes.squeeze()



def read_leadsheet(midi_path, ACC=12):
    """Read a lead sheet MIDI file and return note events in OctMIDI format.
    The MIDI file should have two tracks in the following order: 1) melody, and 2) chords.
    The chord track's register should ideally be around the range of C3-C4.
    """
    midi = pyd.PrettyMIDI(midi_path)
    beat_time = midi.get_beats()
    downbeat_time = midi.get_downbeats()
    beat_time.sort()
    downbeat_time.sort()
    tempo_time, bpm = midi.get_tempo_changes()

    # get a downbeat table [1, 0, 0, 0, 1, 0, 0, ...], where '1' is the downbeat
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
                if (beats_table[idx] > 8) and (beats_table[idx] % 8 == 1):
                    beat_metre[idx] = 1 # ensure that beats_table is in the range of 1 to 8 (i.e., double whole notes at most)
                    beats_table[idx] = 1
            else:
                beats_table[idx] = 0

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

    # get time table for each quantized 1/12 of quarter-beat
    beat_time = np.append(beat_time, beat_time[-1] + (beat_time[-1] - beat_time[-2]))   #extropolate the last beat time
    quantize = interp1d(np.array(range(0, len(beat_time))) * ACC, beat_time, kind='linear') #interploate 1/12 of quarter-beat
    quavers = quantize(np.array(range(0, (len(beat_time) - 1) * ACC)))
    quaver_table = np.zeros(len(beats_table) * ACC)
    for i in range(1, ACC+1):
        quaver_table[i-1::ACC] = (beats_table - 1) * ACC + i - 1
    
    # repeat the downbeats_table and ts_table for each quaver
    downbeats_table = np.repeat(downbeats_table, ACC) - 1
    ts_table = np.repeat(ts_table, ACC, axis=0)
    assert(len(quavers) == len(downbeats_table))

    # quantize the note events
    leadsheet_events = []
    for idx, inst in enumerate(midi.instruments):
        for note in (inst.notes):
            note_start = np.argmin(np.abs(quavers - note.start))
            note_end =  np.argmin(np.abs(quavers - note.end))
            dur = note_end - note_start
            bar = int(downbeats_table[note_start])
            if bar == -1:
                continue
            pos = int(quaver_table[note_start])
            ts = ts_table[note_start]

            current_time = note.start
            current_bpm = bpm[np.argmin([current_time - t for t in tempo_time if current_time - t >= 0])]
            note_event = [current_time,
                            bar, 
                            pos,
                            0 if idx==0 else 4, # melody's instrument is 0, chords' instrument is 4
                            note.pitch,
                            d2e(dur),
                            v2e(note.velocity),
                            t2e((ts, 4)),
                            b2e(current_bpm)
                        ]
            leadsheet_events.append(note_event)

    leadsheet_times = np.array([event[0] for event in leadsheet_events])
    leadsheet_events = np.array([event[1:] for event in leadsheet_events], dtype=int)
    # sort event by bar, position, instrument, and pitch. Pitch is high to low and the other are low to high.
    seq_order = np.lexsort((-leadsheet_events[:, 3], leadsheet_events[:, 2], leadsheet_events[:, 1], leadsheet_events[:, 0]))

    leadsheet_times = leadsheet_times[seq_order]
    leadsheet_events = leadsheet_events[seq_order]

    return downbeat_time, leadsheet_events, leadsheet_times


def get_chord_change_steps(leadsheet_events, leadsheet_times, downbeat_time, STAR_BAR, DURATION):
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
    return chord_change_steps

def add_pedal(pyd_obj, chord_change_steps):
    #pyd_obj is a PrettyMIDI object
    beat_times = pyd_obj.get_beats()
    pedals = []
    for i in range(len(chord_change_steps)-1):
        try:
            pedals.append(pyd.ControlChange(number=64, value=72, time=beat_times[chord_change_steps[i]]))
            pedals.append(pyd.ControlChange(number=64, value=0, time=beat_times[chord_change_steps[i+1]]-0.05))
        except:
            continue
    pyd_obj.instruments[0].control_changes = pedals
    return pyd_obj