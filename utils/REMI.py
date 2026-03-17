import numpy as np


def get_musecoco_attributes():
    """Get a default attribute token list for musecoco, where piano must have, others unknown.
    """
    attributes = {'I1s2_piano': [1, 0, 0], 
            'I1s2_keyboard': [0, 0, 1], 
            'I1s2_percussion': [0, 0, 1], 
            'I1s2_organ': [0, 0, 1], 
            'I1s2_guitar': [0, 0, 1], 
            'I1s2_bass': [0, 0, 1], 
            'I1s2_violin': [0, 0, 1], 
            'I1s2_viola': [0, 0, 1], 
            'I1s2_cello': [0, 0, 1], 
            'I1s2_harp': [0, 0, 1], 
            'I1s2_strings': [0, 0, 1], 
            'I1s2_voice': [0, 0, 1], 
            'I1s2_trumpet': [0, 0, 1], 
            'I1s2_trombone': [0, 0, 1], 
            'I1s2_tuba': [0, 0, 1], 
            'I1s2_horn': [0, 0, 1], 
            'I1s2_brass': [0, 0, 1], 
            'I1s2_sax': [0, 0, 1], 
            'I1s2_oboe': [0, 0, 1], 
            'I1s2_bassoon': [0, 0, 1], 
            'I1s2_clarinet': [0, 0, 1], 
            'I1s2_piccolo': [0, 0, 1], 
            'I1s2_flute': [0, 0, 1], 
            'I1s2_pipe': [0, 0, 1], 
            'I1s2_synthesizer': [0, 0, 1], 
            'I1s2_ethnic_instruments': [0, 0, 1], 
            'I1s2_sound_effects': [0, 0, 1], 
            'I1s2_drum': [0, 0, 1], 
            'R1': [0, 0, 1], 
            'R3': [0, 0, 0, 1], 
            'S2s1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
            'S4_new_age': [0, 0, 1], 
            'S4_electronic': [0, 0, 1], 
            'S4_rap': [0, 0, 1], 
            'S4_religious': [0, 0, 1], 
            'S4_international': [0, 0, 1], 
            'S4_easy_listening': [0, 0, 1], 
            'S4_avant_garde': [0, 0, 1], 
            'S4_rnb': [0, 0, 1], 
            'S4_latin': [0, 0, 1], 
            'S4_children': [0, 0, 1], 
            'S4_jazz': [0, 0, 1], 
            'S4_classical': [0, 0, 1], 
            'S4_comedy_spoken': [0, 0, 1], 
            'S4_pop_rock': [0, 0, 1], 
            'S4_reggae': [0, 0, 1], 
            'S4_stage': [0, 0, 1], 
            'S4_folk': [0, 0, 1], 
            'S4_blues': [0, 0, 1], 
            'S4_vocal': [0, 0, 1], 
            'S4_holiday': [0, 0, 1], 
            'S4_country': [0, 0, 1], 
            'S4_symphony': [0, 0, 1], 
            'B1s1': [0, 0, 0, 0, 1], 
            'TS1s1': [0, 0, 0, 0, 0, 0, 0, 1], 
            'K1': [0, 0, 1], 
            'T1s1': [0, 0, 0, 1], 
            'P4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
            'EM1': [0, 0, 0, 0, 1], 
            'TM1': [0, 0, 0, 0, 0, 1]
            }
    
    labels_I1s2 = []
    labels_S4 = []
    for key in list(attributes.keys()):
        if key[:4] == 'I1s2':
            labels_I1s2.append(attributes[key])
            attributes.pop(key)
        if key[:2] == 'S4':
            labels_S4.append(attributes[key])
            attributes.pop(key)
    attributes['I1s2'] = labels_I1s2
    attributes['S4'] = labels_S4

    attribute_tokens = []
    for k in attributes:
        if k == 'I1s2' or k == 'S4':
            for i in range(len(attributes[k])):
                attribute_tokens.append(f"{k}_{i}_{np.argmax(attributes[k][i])}")
        else:
            v = attributes[k]
            attribute_tokens.append(f"{k}_{np.argmax(v)}")
    attribute_tokens.append('<sep>')

    return attribute_tokens


def note2token(track, redact_performance=False):
    """Convert note attribute indices to token strings for musecoco."""
    token_list = []
    last_bar = -1
    last_onset = -1
    last_inst = -1
    for i, event in enumerate(track):
        if event[0] != last_bar:
            token_list.append('b-1')
            token_list.append(f's-{event[6]}')
            if not redact_performance:
                token_list.append(f't-{event[7]}')
            else:
                token_list.append(f'<unk>')
            last_bar = event[0]
            last_onset = -1
            last_inst = -1
        if event[1] != last_onset:
            token_list.append(f'o-{event[1]}')
            last_onset = event[1]
            last_inst = -1
        if event[2] != last_inst:
            token_list.append(f'i-{event[2]}')
            last_inst = event[2]
        token_list.append(f'p-{event[3]}')
        token_list.append(f'd-{event[4]}')
        if not redact_performance:
            token_list.append(f'v-{event[5]}')
        else:
            token_list.append(f'<unk>')
    token_list = token_list[1:] + token_list[:1]
    return token_list