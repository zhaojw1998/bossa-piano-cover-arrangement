import os
import pretty_midi as pyd
import numpy as np
from scipy.interpolate import interp1d


from octMIDI import t2e, b2e, d2e, v2e, encoding_to_MIDI, encoding_to_str, CHECKTABLE
import re

from tqdm import tqdm

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import glob
import shutil

#import sys
#sys.path.append("/home/zhaojw/chord_recognition/")
#from main import transcribe_cb1000_midi

import miditoolkit
from miditoolkit.midi.containers import Marker
import copy
import mir_eval
import h5py



def read_chord(markers):
    notes = []
    for idx in range(len(markers)-1):
        if 'N' in markers[idx].text:
            continue

        root, quality, scale_degree, bass = mir_eval.chord.split(markers[idx].text)
        bitmap = mir_eval.chord.quality_to_bitmap(quality)
        for degree in scale_degree:
            bitmap[mir_eval.chord.scale_degree_to_semitone(degree) % 12] = 1
            #print(bitmap)
        #print('bass', bass)
        root = mir_eval.chord.pitch_class_to_semitone(root)
        bass = (mir_eval.chord.scale_degree_to_semitone(bass) + root) % 12
        bitmap = np.roll(bitmap, root-bass)

        if bass + np.argmax(bitmap) > 11:
            register = 3
        else:
            register = 4
        for semitone in np.flatnonzero(bitmap):
            notes.append(pyd.Note(pitch=register*12+bass+semitone,
                            start=markers[idx].time,
                            end=markers[idx+1].time,
                            velocity=80))
    return notes




triple_meter_pieces = ['034', '062', '102', '107', '152', '173', '176', '203', '215', '231', '254', '280', '307', '328', '369', '584', '592', '624', '653', '654', '662', '744', '749', '756', '770', '799', '869', '872', '887']
        
triple_quaver_pieces = ['092', '171', '311', '350', '360', '379', '393', '412', '509', '575', '579', '632', '678', '689', '693', '741', '775', '801', '806', '843', '856']


midi_dir = "/data1/zhaojw/Q&A/POP909-Dataset/POP909/"
save_root = "/data1/zhaojw/Q&A/POP909-Dataset/pop909_midi_12bins/"
ACC = 12

cnt=0
total=10
for song_name in tqdm(os.listdir(midi_dir)):
    if song_name == 'index.xlsx':
                continue
    midi = pyd.PrettyMIDI(os.path.join(midi_dir, song_name, f'{song_name}.mid'))

    beats = np.loadtxt(os.path.join(midi_dir, song_name, 'beat_midi.txt'))
    beat_time, simple_metre, compound_metre = beats[:, 0], beats[:, 1], beats[:, 2]

    with open(os.path.join(midi_dir, song_name, 'chord_midi.txt')) as f:
        lines = f.readlines()
    chords = []
    for line in lines:
        start, end, chord_label = line.replace('\n', '').split('\t')
        chords.append(Marker(chord_label, float(start)))

    # read tempo changes
    tempo_time, bpm = midi.get_tempo_changes()

    # get beat metre
    if song_name in triple_meter_pieces:    #3/4 time signature
        beat_metre = simple_metre
    else:   #4/4 time signature
        beat_metre = compound_metre
    downbeat_time = beat_time[beat_metre == 1]

    # get a beat table [1, 2, 3, 4, 1, 2, 3, ...], where '1' is the downbeat
    beats_table = np.zeros(len(beat_metre))
    for idx, value in enumerate(beat_metre):
        if value == 1:
            beats_table[idx] = 1
        elif value == 0:
            if beats_table[idx - 1] > 0:
                beats_table[idx] = beats_table[idx - 1] + 1
                if (beats_table[idx] > 8) and (beats_table[idx] % 8 == 1):
                    beat_metre[idx] = 1
                    beats_table[idx] = 1
            else:
                beats_table[idx] = 0

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

    note_events = []
    for inst in midi.instruments:
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
            ts = ts_table[note_start]

            current_time = note.start#quavers[note_start]
            current_bpm = bpm[np.argmin([current_time - t for t in tempo_time if current_time - t >= 0])]
            note_event = [current_time,\
                            bar, \
                            pos,\
                            128 if inst.is_drum else inst.program,\
                            128+note.pitch if inst.is_drum else note.pitch,\
                            d2e(dur),\
                            v2e(note.velocity),\
                            t2e((ts, 4)),\
                            b2e(current_bpm)
                        ]
            note_events.append(note_event)
    note_events.sort()

    note_times = np.array([event[0] for event in note_events])
    note_events = np.array([event[1:] for event in note_events], dtype=int)


    leadsheet_events = []
    for note in (midi.instruments[0].notes):
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

        current_time = note.start#quavers[note_start]
        current_bpm = bpm[np.argmin([current_time - t for t in tempo_time if current_time - t >= 0])]
        note_event = [current_time,\
                        bar, \
                        pos,\
                        128 if inst.is_drum else inst.program,\
                        128+note.pitch if inst.is_drum else note.pitch,\
                        d2e(dur),\
                        v2e(note.velocity),\
                        t2e((ts, 4)),\
                        b2e(current_bpm)
                    ]
        leadsheet_events.append(note_event)

    for note in (read_chord(chords)):
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

        current_time = note.start#quavers[note_start]
        current_bpm = bpm[np.argmin([current_time - t for t in tempo_time if current_time - t >= 0])]
        note_event = [current_time,\
                        bar, \
                        pos,\
                        4,\
                        note.pitch,\
                        d2e(dur),\
                        v2e(note.velocity),\
                        t2e((ts, 4)),\
                        b2e(current_bpm)
                    ]
        leadsheet_events.append(note_event)
    leadsheet_events.sort()

    leadsheet_times = np.array([event[0] for event in leadsheet_events])
    leadsheet_events = np.array([event[1:] for event in leadsheet_events], dtype=int)


    """midi_recon = encoding_to_MIDI(note_events)
    ls_recon = encoding_to_MIDI(leadsheet_events)
    midi_recon.instruments += ls_recon.instruments
    midi_recon.dump(os.path.join(save_root, f'{song_name}_test_recon.mid'))
    shutil.copy(os.path.join(midi_dir, song_name, f'{song_name}.mid'), os.path.join(save_root, f'{song_name}_test_origin.mid'))"""


    np.savez(os.path.join(save_root, f'{song_name}.npz'),
                 note_times=note_times, 
                 note_events=note_events, 
                 leadsheet_times=leadsheet_times, 
                 leadsheet_events=leadsheet_events,
                 downbeat_times=downbeat_time
                 )
    #cnt += 1
    #if cnt >= total:
    #    break

