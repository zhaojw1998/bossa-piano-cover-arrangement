import os
import sys
from . import midiprocessor as mp
import pretty_midi

#from utils_chord.chord_map import recognize_chord_from_midi, quantize_chord
from .midi_data_extractor.utils.pos_process import fill_pos_ts_and_tempo_
# from .midi_data_extractor.utils.pos_process import fill_pos_ts_and_tempo_
from .midi_data_extractor.midi_processing import get_midi_pos_info, convert_pos_info_to_tokens
from .midi_data_extractor.data_extractor import get_bar_positions, get_bars_insts, ChordDetector
#from utils_texture.texture_tools import get_time_function_from_remi_one_bar

from .midiprocessor import MidiDecoder

ls = os.listdir
jpath = os.path.join


def _main():
    # test_remi_tok()
    check_chord_recognizer()
    # check_jingwei_chord_recognizer()


def test_remi_tok():
    dataset_dir = 'datasets/local/midis'
    output_dir = 'datasets/local/midis_recovered'
    remi_fp = 'remi_test.txt'
    midi_fns = ls(dataset_dir)
    midi_fns.sort()
    tk = RemiTokenizer()

    # remis = []
    # for midi_fn in midi_fns:
    #     midi_fp = jpath(dataset_dir, midi_fn)
    #     remi_toks = tk.midi_to_remi(midi_fp)
    #     remi_tok_strs = ' '.join(remi_toks)
    #     remis.append(remi_tok_strs + '\n')
    #
    # out_fp = remi_fp
    # with open(out_fp, 'w') as f:
    #     f.writelines(remis)

    with open(remi_fp) as f:
        remis = f.readlines()
    for id, remi in enumerate(remis):
        remi = remi.split(' ')
        output_fp = jpath(output_dir, 'midi_{}.mid'.format(id + 1))
        tk.remi_to_midi(remi, output_fp)


def check_chord_recognizer():
    '''
    Check Magenta's script to recognize chords from MIDI
    '''
    # midi_fp = './datasets/local/segmented/Track01876-3/midi.mid'
    midi_fp = './datasets/local/segmented/Track01501-1/midi.mid'
    tk = RemiTokenizer()
    tk.recognize_chord_from_midi(midi_fp)


# def check_jingwei_chord_recognizer():
#     from chord_recognition.main import transcribe_cb1000_midi
#     # midi_fp = './datasets/local/midis/test2.mid'
#     midi_fp = './datasets/local/segmented/Track01501-1/midi.mid'
#     transcribe_cb1000_midi(midi_fp, 'midi_chord.txt')


def _procedures():
    pass


def get_midi_length_in_seconds(midi_file_path):
    import mido

    mid = mido.MidiFile(midi_file_path)

    # 设置默认tempo（500000微秒每拍，即120拍每分钟）
    current_tempo = 500000
    midi_length = 0

    for track in mid.tracks:
        elapsed_time = 0
        for msg in track:
            # 如果是tempo事件，更新当前tempo
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
            elapsed_time += msg.time

        # 使用当前tempo将ticks时间转换为秒
        seconds = mido.tick2second(elapsed_time, mid.ticks_per_beat, current_tempo)
        midi_length = max(midi_length, seconds)

    return midi_length


class MidiUtil:
    @staticmethod
    def get_duration(midi_fp):
        midi_data = pretty_midi.PrettyMIDI(midi_fp)
        length_in_seconds = midi_data.get_end_time()
        return length_in_seconds


class RemiTokenizer:
    def __init__(self):
        self.encoding_method = "REMIGEN2"
        self.midi_encoder = mp.MidiEncoder(self.encoding_method)
        self.midi_decoder = MidiDecoder(self.encoding_method)
        self.chord_detector = ChordDetector(self.midi_encoder)

        self.attribute_list = ['I1s2', 'R1', 'R3', 'S2s1', 'S4', 'B1s1', 'TS1s1', 'K1', 'T1s1', 'P4', 'EM1', 'TM1']

    def midi_to_remi(self, midi_path, key_normalize=True, return_pitch_shift=False, return_key=False):
        '''
        Convert a midi file to a remi sequence
        :param midi_path:
        :return: a list of str, remi token sequence.
        '''
        midi_obj = mp.midi_utils.load_midi(midi_path)
        #midi_obj.instruments = midi_obj.instruments[0:1]
        pos_info = get_midi_pos_info(self.midi_encoder, midi_path=None, midi_obj=midi_obj, remove_empty_bars=False)
        # Longshen: don't remove empty bars to facilitate

        pos_info = fill_pos_ts_and_tempo_(pos_info)

        # Normalize pitch to C maj or A min.
        is_major = None
        normalize_pitch_value = key_normalize
        if normalize_pitch_value:
            try:
                pos_info, is_major, pitch_shift = self.midi_encoder.normalize_pitch(pos_info)
                # Pitch shift is an integer number which directly add towards the pitch of all notes inside pos_info
            except KeyboardInterrupt:
                raise
            except:
                is_major = None

        # Get bar token index, and instruments inside each bar
        bars_positions = get_bar_positions(pos_info)
        bars_instruments = get_bars_insts(pos_info, bars_positions)
        num_bars = len(bars_positions)
        assert num_bars == len(bars_instruments)

        # Get the tokens
        tokens = convert_pos_info_to_tokens(
            self.midi_encoder,
            pos_info,
            ignore_ts=False,
            ignore_tempo=False,
            ignore_velocity=False,
        )
        assert tokens[-1] == 'b-1'

        if return_pitch_shift:
            if return_key:
                return tokens, pitch_shift, is_major
            return tokens, pitch_shift
        else:
            return tokens

    def get_bar_pos(self, tokens):
        '''
        Get the index of starting tokens of each bar, from remi tokens.
        Specifically used for data preparation.

        :param tokens:
        :return:
        '''
        # Get the starting token of each bar
        start_token_index_of_the_bar = 0
        bar_id = 0
        bar_start_token_indices = {}
        # bars_token_positions[bar_id] = (start token index of this bar, start token index of next bar)
        for idx, token in enumerate(tokens):
            if token == 'b-1':
                start_token_index_of_next_bar = idx + 1
                bar_start_token_indices[bar_id] = (start_token_index_of_the_bar, start_token_index_of_next_bar)

                # Go to the next bar
                start_token_index_of_the_bar = start_token_index_of_next_bar
                bar_id = bar_id + 1
        return bar_start_token_indices

    def remi_to_midi(self, remi_seq, midi_path=None, ignore_velocity=True):
        '''
        Convert a list of remi tokens to a midi and save to disk.
        :param remi_seq: a list of str, each of which represent remi tokens.
        :param midi_path:
        :return:
        '''
        midi_obj = self.midi_decoder.decode_from_token_str_list(remi_seq, ignore_velocity=ignore_velocity)
        if midi_path is not None:
            midi_obj.dump(midi_path)
        return midi_obj

    # @classmethod
    # def remi_to_midi(cls, remi_seq, midi_path):
    #     '''
    #     Convert a list of remi tokens to a midi and save to disk.
    #     :param remi_seq: a list of str, each of which represent remi tokens.
    #     :param midi_path:
    #     :return:
    #     '''
    #     midi_obj = cls.midi_decoder.decode_from_token_str_list(remi_seq)
    #     midi_obj.dump(midi_path)

    def midi_to_remi_file(self, midi_path, remi_path, return_pitch_shift=False):
        if return_pitch_shift:
            remi_seq, pitch_shift = self.midi_to_remi(midi_path, return_pitch_shift)
        else:
            remi_seq = self.midi_to_remi(midi_path, return_pitch_shift)
            pitch_shift = None
        remi_tok_strs = ' '.join(remi_seq)
        with open(remi_path, 'w') as f:
            f.write(remi_tok_strs + '\n')

        if return_pitch_shift:
            return pitch_shift

    def recognize_chord_from_midi_old(self, midi_path):
        '''
        Method by MuseCoco. Not recommended due to inaccurate detection.

        Recognize chord info from midi file using Magenta's script
        :param midi_path: path of midi file.
        :return: a list of string, recognized chords
        '''
        midi_obj = mp.midi_utils.load_midi(midi_path)
        pos_info = get_midi_pos_info(self.midi_encoder, midi_path=None, midi_obj=midi_obj, remove_empty_bars=False)
        # Longshen: don't remove empty bars to facilitate
        pos_info = fill_pos_ts_and_tempo_(pos_info)
        # pos_info, is_major, _ = self.midi_encoder.normalize_pitch(pos_info)
        bars_chords = self.chord_detector.infer_chord_for_pos_info(pos_info)
        # print(bars_chords)
        # exit(10)
        return bars_chords


class RemiUtil:
    def __init__(self):
        pass

    @staticmethod
    def extract_condition_from_remi(remi_seq, key, dir):
        '''
        Dir: path to save the segment midis
        '''
        bar_indices = RemiUtil.get_bar_idx_from_remi(remi_seq)
        num_bars = len(bar_indices)
        tk = RemiTokenizer()

        # print(num_bars)
        if num_bars == 0:
            raise Exception('Bar num = 0')

        all_bar_info = {}
        for bar_id in bar_indices:
            bar_start_idx, bar_end_idx = bar_indices[bar_id]
            bar_remi = remi_seq[bar_start_idx:bar_end_idx]
            bar_info = {}
            bar_info['remi_seq'] = bar_remi

            # Detokenize the segment
            midi_seg_fp = jpath(dir, f'bar_{bar_id}.mid')
            tk.remi_to_midi(bar_remi, midi_seg_fp)

            # Key and scale
            key_segment = key
            bar_info['is_major'] = key_segment
            bar_info['scale'] = 1

            # Instruments
            insts_this_bar = set()
            for token in bar_remi:
                if token.startswith('i-'):
                    insts_this_bar.add(token)
            insts_this_bar = list(insts_this_bar)
            insts_this_bar = sorted(insts_this_bar, key=lambda x: int(x.split('-')[1]))  # sort by inst id
            bar_info['instruments'] = insts_this_bar

            # Recognize chord
            chords = recognize_chord_from_midi(midi_seg_fp, out_fp=None)
            dur = MidiUtil.get_duration(midi_seg_fp)
            chord_quantized = quantize_chord(chords, dur=dur, num_bars=1, num_chord_per_bar=2).tolist()
            bar_info['chords'] = chord_quantized

            # Obtain time function
            txt_seq = get_time_function_from_remi_one_bar(bar_remi)
            bar_info['onset_density'] = txt_seq

            all_bar_info[bar_id] = bar_info

        return all_bar_info

    @staticmethod
    def get_bar_idx_from_remi(remi_seq):
        # Get the starting token of each bar
        start_token_index_of_the_bar = 0
        bar_id = 0
        bar_indices = {}

        # bars_token_positions[bar_id] = (start token index of this bar, start token index of next bar)
        for idx, token in enumerate(remi_seq):
            if token == 'b-1':
                start_token_index_of_next_bar = idx + 1
                bar_indices[bar_id] = (start_token_index_of_the_bar, start_token_index_of_next_bar)

                # Go to the next bar
                start_token_index_of_the_bar = start_token_index_of_next_bar
                bar_id = bar_id + 1
        return bar_indices

    @staticmethod
    def get_inst_from_remi(remi_seq):
        '''
        Get a list of instrument tokens from a sequence of remi tokens, for each bar of info
        There is no repeated instrument in the list.
        Each list is sorted by program id.
        e.g., [
                [i-0, i-8, i-128], # insts of 1st bar
                [i-0, i-128],    # insts of 2nd bar
            ]
        '''
        b_1_indices = [index for index, element in enumerate(remi_seq) if element == "b-1"]
        num_bars = len(b_1_indices)
        b_1_indices.insert(0, -1)

        insts_each_bar = []
        for bar_id in range(num_bars):
            bar_start_idx = b_1_indices[bar_id] + 1
            bar_end_idx = b_1_indices[bar_id + 1]
            bar_seq = remi_seq[bar_start_idx:bar_end_idx]

            # Obtain instruments from bar_seq
            insts_this_bar = set()
            for token in bar_seq:
                if token.startswith('i-'):
                    insts_this_bar.add(token)
            insts_this_bar = list(insts_this_bar)
            insts_this_bar = sorted(insts_this_bar, key=lambda x: int(x.split('-')[1]))  # sort by inst id

            # Add instrument info to input_tokens
            insts_each_bar.append(insts_this_bar)

        return insts_each_bar

    @staticmethod
    def get_inst_from_input_seq(inp_seq):
        input_list = inp_seq

        # 初始化结果列表和临时子列表
        result = []
        current_sublist = []

        for item in input_list:
            if item.startswith('i-'):
                # 如果遇到以'i-'开头的元素，将它加入sub list
                current_sublist.append(item)
            else:  # 如果元素不以i开头，那么将sub list加入result
                if len(current_sublist) > 0:
                    result.append(current_sublist)
                    current_sublist = []

        # 循环结束后，如果当前子列表非空，将其添加到结果中
        if len(current_sublist) > 0:
            result.append(current_sublist)

        return result

    @staticmethod
    def get_chord_from_input_seq(input_seq):
        '''
        Get chord info from input sequence
        CR-X, CT-Y
        :param input_seq:
        :return: Two lists, first is chord root, second is chord type.
        e.g., [[CR-0, CR-2], [CR-1, CR-1]],    [[CT-5, CT-7], [CT-5, CT-0]]
        '''
        # 初始化结果列表和临时子列表
        res_cr = []
        res_ct = []

        # chord_root_cur_bar = []
        # chord_type_cur_bar = []
        for token in input_seq:
            if token.startswith('CR'):
                # 如果遇到以'i-'开头的元素，将它加入sub list
                # chord_root_cur_bar.append(token)
                res_cr.append(token)
            elif token.startswith('CT'):
                # chord_type_cur_bar.append(token)
                res_ct.append(token)
            # elif token == 'b-1':
            #     res_cr.append(chord_root_cur_bar)
            #     res_ct.append(chord_type_cur_bar)
            #     chord_root_cur_bar = []
            #     chord_type_cur_bar = []

        return res_cr, res_ct

    @staticmethod
    def get_rhythm_from_input_seq(inp_seq):
        '''
        Get the onset density on each position, from the input sequence.
        :param inp_seq: input condition token sequence. May contain info for several bars.
        :return: a list of dict, each dict is the rhythm info inside a bar. Key is position, value is onset count.
            [
                {
                    2: 8, 3, 4,
                },
                {
                    0: 10, 8: 5,
                }
            ]
        '''
        res = []
        for tok in inp_seq:
            if tok == 'TF':
                t = {}
            elif tok == 'b-1':
                res.append(t)
            elif tok.startswith('o-'):
                cur_pos = int(tok.split('-')[-1])
            elif tok.startswith('TF-'):
                onset_cnt = int(tok.split('-')[-1])
                t[cur_pos] = onset_cnt
        return res

    @staticmethod
    def read_remi(fp, split=True, remove_input=False):
        with open(fp) as f:
            remi_str = f.readline().strip()

        if remove_input:
            remi_str = remi_str.split(' <sep> ')[1]

        if split:
            remi_seq = remi_str.split(' ')
        else:
            remi_seq = remi_str
        return remi_seq

    @staticmethod
    def save_remi_seq(remi_seq, fp):
        with open(fp, 'w') as f:
            f.write(' '.join(remi_seq))


if __name__ == '__main__':
    _main()
