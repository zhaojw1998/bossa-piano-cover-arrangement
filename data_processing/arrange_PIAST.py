import os
import pretty_midi as pyd
import numpy as np
from scipy.interpolate import interp1d
from miditoolkit.midi.containers import Marker
import miditoolkit
import mir_eval
import h5py
from tqdm import tqdm

from octMIDI import t2e, b2e, d2e, v2e, encoding_to_MIDI, encoding_to_str, CHECKTABLE

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)



def read_chord(markers, pyd_object):
    notes = []
    for idx in range(len(markers)-1):
        if 'bpm' in markers[idx].text:
            continue
        if 'N' in markers[idx].text:
            continue
        #try:
        root, quality, bass = markers[idx].text.split('_')
        if 'M' in quality:
            quality = 'maj' + quality[1:]
        elif 'm' in quality:
            quality = 'min' + quality[1:]
        elif '+' in quality:
            quality = 'aug' + quality[1:]
        elif '/o' in quality:
            quality = 'hdim7'
        elif 'o' in quality:
            quality = 'dim' + quality[1:]
        try:
            chord_label = f'{root}:{quality}'
            mir_eval.chord.validate_chord_label(chord_label)
        except:
            print(markers[idx], chord_label)
            import sys
            sys.exit()

        root = mir_eval.chord.pitch_class_to_semitone(root)
        quality = mir_eval.chord.quality_to_bitmap(quality)
        bass = mir_eval.chord.pitch_class_to_semitone(bass)

        quality = np.roll(quality, root)
        bitmap = np.roll(quality, -bass)
        bitmap[0] = 1
        for semitone in np.flatnonzero(bitmap):
            notes.append(pyd.Note(pitch=4*12+bass+semitone,
                            start=pyd_object.tick_to_time(markers[idx].time),
                            end=pyd_object.tick_to_time(markers[idx+1].time),
                            velocity=80))
    return notes


audio_dir = "/data1/zhaojw/PIAST/audio_data.h5"
with h5py.File(audio_dir, "r") as f:
    data_to_load =list(f.keys())
#import random
#random.shuffle(data_to_load)
midi_dir = "/data1/zhaojw/PIAST/piast_yt/midi/"
save_root = "/data1/zhaojw/PIAST/piast_midi_12bins/"
ACC = 12

for song_name in tqdm(data_to_load):
    try:
        midi = pyd.PrettyMIDI(os.path.join(midi_dir, f'{song_name}.mid'))
        midi_obj = miditoolkit.midi.parser.MidiFile(os.path.join(midi_dir, f'{song_name}.mid'))
        midi_obj.markers.append(Marker(text='N_N_N', time=midi_obj.max_tick))
        chord_notes = read_chord(midi_obj.markers, midi)
    except:
        continue

    # read time signature changes
    midi.time_signature_changes = []    #time signature is not reliable
    # read beat/downbeat, and time signature from midi
    beat_time = midi.get_beats()
    downbeat_time = midi.get_downbeats()
    beat_metre = np.zeros(len(beat_time))
    for i, bt in enumerate(beat_time):
        if (bt in downbeat_time):
            beat_metre[i] = 1
    assert(sum(beat_metre)==len(downbeat_time))
    # read tempo changes
    tempo_time, bpm = midi.get_tempo_changes()
    #print(tempo_time, bpm)
    
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

    """ts_table = np.ones(len(beat_time))
    last = 0
    highest = 0
    for idx, numerator in enumerate(beats_table[::-1]):
        if numerator >= last:
            ts_table[-1-idx] = numerator
            highest = numerator
        else:
            ts_table[-1-idx] = highest
        last = numerator"""

    # get time table for each quantized 1/16 of quarter-beat
    beat_time = np.append(beat_time, beat_time[-1] + (beat_time[-1] - beat_time[-2]))
    quantize = interp1d(np.array(range(0, len(beat_time))) * ACC, beat_time, kind='linear')
    quavers = quantize(np.array(range(0, (len(beat_time) - 1) * ACC)))
    quaver_table = np.zeros(len(beats_table) * ACC)
    for i in range(1, ACC+1):
        quaver_table[i-1::ACC] = (beats_table - 1) * ACC + i - 1
    downbeats_table = np.repeat(downbeats_table, ACC) - 1
    #ts_table = np.repeat(ts_table, ACC)
    assert(len(quavers) == len(downbeats_table))

    note_events = []
    for itk, inst in enumerate(midi.instruments):
        for note in inst.notes:
            if note.pitch < 6 or note.pitch >= 122:
                continue
            note_start = np.argmin(np.abs(quavers - note.start))
            note_end =  np.argmin(np.abs(quavers - note.end))
            dur = note_end - note_start
            bar = int(downbeats_table[note_start])
            if bar == -1:
                continue
            pos = int(quaver_table[note_start])
            #ts = ts_table[note_start]
            current_time = note.start#quavers[note_start]
            current_bpm = bpm[np.argmin([current_time - t for t in tempo_time if current_time - t >= 0])]
            note_event = [current_time,\
                            bar, \
                            pos,\
                            itk,\
                            note.pitch,\
                            d2e(dur),\
                            v2e(note.velocity),\
                            t2e((4, 4)),\
                            b2e(current_bpm)
                        ]
            note_events.append(note_event)

    for note in chord_notes:
        note_start = np.argmin(np.abs(quavers - note.start))
        note_end =  np.argmin(np.abs(quavers - note.end))
        dur = note_end - note_start
        if dur == 0:
            continue
        bar = int(downbeats_table[note_start])
        if bar == -1:
            continue
        pos = int(quaver_table[note_start])
        #ts = int(ts_table[note_start])
        current_time = quavers[note_start]
        current_bpm = bpm[np.argmin([current_time - t for t in tempo_time if current_time - t >= 0])]
        
        note_event = [current_time,\
                        bar, \
                        pos,\
                        2,\
                        note.pitch,\
                        d2e(dur),\
                        v2e(note.velocity),\
                        t2e((4, 4)),\
                        b2e(current_bpm),\
                    ]
        note_events.append(note_event)
    note_events.sort()


    note_times = np.array([event[0] for event in note_events])
    note_events = np.array([event[1:] for event in note_events], dtype=int)

    leadsheet_times = note_times[note_events[:, 2]!=0]
    leadsheet_events = note_events[note_events[:, 2]!=0]
    leadsheet_events[leadsheet_events[:, 2]==1, 2] = 0  #melody
    leadsheet_events[leadsheet_events[:, 2]==2, 2] = 4  #chord

    note_times = note_times[note_events[:, 2]==0]
    note_events = note_events[note_events[:, 2]==0]

    np.savez(os.path.join(save_root, f'{song_name}.npz'),
                note_times=note_times, 
                note_events=note_events, 
                leadsheet_times=leadsheet_times, 
                leadsheet_events=leadsheet_events,
                downbeat_times=downbeat_time
                )
    #break

