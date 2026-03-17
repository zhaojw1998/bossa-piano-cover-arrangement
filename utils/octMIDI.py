import re
import math
import miditoolkit
import numpy as np


pos_resolution = 12  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1000  # window length max
sample_overlap_rate = 4
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127


ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))

dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]

def e2t(x):
    return ts_list[x]

def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]

def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]

def v2e(x):
    return x // velocity_quant

def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)

def b2e(x):
    x = np.clip(x, min_tempo, max_tempo)
    x = x / min_tempo
    e = np.round(np.log2(x) * tempo_quant)
    return e

def e2b(x):
    return 2 ** (x / tempo_quant) * min_tempo

def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator

def MIDI_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                             1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    encoding.sort()
    return encoding

def encoding_to_MIDI(encoding):
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * pos_resolution#ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, e2d(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = e2v(i[5])
        midi_obj.instruments[program].notes.append(miditoolkit.Note(
            start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj

def str_to_encoding(s):
    encoding = [int(i[3: -1]) for i in s.split() if 's' not in i]
    tokens_per_note = 8
    assert len(encoding) % tokens_per_note == 0
    encoding = [tuple(encoding[i + j] for j in range(tokens_per_note))
                for i in range(0, len(encoding), tokens_per_note)]
    return encoding

def encoding_to_str(e, bar_index_offset=0):
    p = 0
    tokens_per_note = 8
    return ' '.join((['<s>'] * tokens_per_note)
                    + ['<{}-{}>'.format(j, int(k) if j > 0 else k + bar_index_offset) for i in e if i[0] + bar_index_offset < bar_max for j, k in enumerate(i)]
                    + (['</s>'] * (tokens_per_note)))

def get_dictionary():
    dictionary = {}
    dictionary['<s>'] = 0
    dictionary['<pad>'] = 1
    dictionary['</s>'] = 2
    dictionary['<unk>'] = 3

    for bar_idx in range(0, 256):
        dictionary[f'<0-{bar_idx}>'] = len(dictionary)  #0.25
    for pos_idx in range(0, 128):
        dictionary[f'<1-{pos_idx}>'] = len(dictionary)  #0.25
    for inst_idx in range(0, 129):
        dictionary[f'<2-{inst_idx}>'] = len(dictionary)  #0.05
    for pitch_idx in range(0, 256):
        dictionary[f'<3-{pitch_idx}>'] = len(dictionary)  #0.25
    for dur_idx in range(0, 128):
        dictionary[f'<4-{dur_idx}>'] = len(dictionary)  #0.1
    for vel_idx in range(0, 32):
        dictionary[f'<5-{vel_idx}>'] = len(dictionary)  #0.05
    for ts_idx in range(0, 254):
        dictionary[f'<6-{ts_idx}>'] = len(dictionary)   #0.025
    for tempo_idx in range(0, 49):
        dictionary[f'<7-{tempo_idx}>'] = len(dictionary)    #0.025
    
    dictionary['<mask>'] = len(dictionary)
    return dictionary

CHECKTABLE = get_dictionary()


def note_encode(note_attribtues, bar_index_offset=0):
    """Convert OctMIDI attribtues to token indices for training."""
    string = encoding_to_str(note_attribtues, bar_index_offset)
    SPACE_NORMALIZER = re.compile(r"\s+")
    string = SPACE_NORMALIZER.sub(" ", string)
    string = string.strip()
    string = string.split()
    return np.array([CHECKTABLE[st] for st in string])



if __name__ == '__main__':
    import miditoolkit
    midi_path = 'mirex/mirex_kevin.mid'
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    encoding = MIDI_to_encoding(midi_obj)
    encoding = [enc for enc in encoding if enc[0] < 2]
    encoding = [[*enc] for enc in encoding]
    encoding = np.array(encoding, dtype=int)
    prompt = encoding
    import sys
    sys.path.append('/data2/zhaojw/a2s_style_learning')
    from musecoco.hf_musecoco.tokenization_musecoco import MuseCocoTokenizer
    from musecoco.hf_musecoco.midi_utils.midiprocessor.vocab_manager import VocabManager
    #from data_processing.octMIDI import e2t, e2d, e2v, e2b, note_encode
    from utils.REMI import note2token
    VM = VocabManager()
    from dataset import collate_fn_inference, TK

    prompt[prompt[:, 2]==128, 3] = np.array([VM.convert_pitch_to_id(pitch-128, True) for pitch in prompt[prompt[:, 2]==128, 3]])
    prompt[prompt[:, 2]!=128, 3] = np.array([VM.convert_pitch_to_id(pitch, False) for pitch in prompt[prompt[:, 2]!=128, 3]])
    prompt[:, 4] = np.array([VM.convert_dur_to_id(e2d(dur)) for dur in prompt[:, 4]])
    prompt[:, 5] = np.array([VM.convert_vel_to_id(e2v(tempo)) for tempo in prompt[:, 5]])
    prompt[:, 6] = np.array([VM.convert_ts_to_id(e2t(ts)) for ts in prompt[:, 6]])
    prompt[:, 7] = np.array([VM.convert_tempo_to_id(e2b(bpm)) for bpm in prompt[:, 7]])
    prompt = note2token(prompt)
    prompt = TK(' '.join(prompt), return_tensors="pt")['input_ids'][:, :-1]#.to(DEVICE)
    print(prompt)