import numpy as np
import torch
from tqdm import tqdm


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


def generate_nucleus(musecoco, inp, t, p=None, k=None):
    # print('-----------------')
    # print(samples["text_input"])
    # print(samples["text_output"])
    # print('-----------------')
    
    decoder_input_ids = inp
    for _ in tqdm(range(1024)):
        if _ == 0:
            outputs = musecoco(
                input_ids=decoder_input_ids,
                use_cache=True,
                return_dict=True
            )
        else:
            outputs = musecoco(
                input_ids=decoder_input_ids[:, -1:],
                use_cache=True,
                return_dict=True,
                past_key_values=outputs.past_key_values
            )

        #print(decoder_input_ids)
        logits = outputs.logits[:, -1] 
        #monotonic_mask = monosampler.get_sample_mask(decoder_input_ids[0,-1]).to(inp.device)
        #logits = logits.masked_fill(~monotonic_mask, float('-inf'))
        if p is not None:
            logits = nucleus_filter(logits / t, p)
        elif (p is None) and (k is not None):
            zeros = logits.new_ones(logits.shape) * float('-inf')
            values, indices = torch.topk(logits, k, dim=-1)
            logits = zeros.scatter(-1, indices, values/t)

        probability = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.multinomial(probability, 1)
        if prediction[0] == 2:
            break
        decoder_input_ids = torch.cat([decoder_input_ids, prediction], dim=-1)

    return decoder_input_ids



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


if __name__ == '__main__':

    import os
    from hf_musecoco.modeling_musecoco import MuseCocoLMHeadModel
    from hf_musecoco.configuration_musecoco import MuseCocoConfig
    from hf_musecoco.tokenization_musecoco import MuseCocoTokenizer
    from hf_musecoco.midi_utils.utils_midi import RemiTokenizer

    #select model
    MODEL_SIZE = '1b'
    PRETRAINED_MUSECOCO_PATH = './'

    #load tokenizer
    tk_fp = os.path.join(PRETRAINED_MUSECOCO_PATH, MODEL_SIZE, 'tokenizer')
    TK = MuseCocoTokenizer.from_pretrained(tk_fp)
    
    #load model
    model_fp = os.path.join(PRETRAINED_MUSECOCO_PATH, MODEL_SIZE, 'model')
    model = MuseCocoLMHeadModel.from_pretrained(model_fp)
    model.cuda()

    #prepare attribute (condition) tokens
    attribute_tokens = get_musecoco_attributes()  #fixed to piano only
    token_list = ['</s>'] + attribute_tokens

    #load midi and convert to token
    midi_tok = RemiTokenizer()
    midi_path = "midi_path.mid"
    remi = midi_tok.midi_to_remi(midi_path, key_normalize=False)
    remi = remi[:200]   #take the first 200 tokens, and continue generating from there
    
    token_list += remi
    tokens = [] #batch of tokens
    tokens.append(' '.join(token_list))
    tokens = TK(tokens, return_tensors="pt")['input_ids'][:, :-1]   #remove the last token, which is </s>

    #model inference
    out = generate_nucleus(model, tokens, t=1, p=None, k=15)

    #convert output to midi
    out_str = TK.batch_decode(out)
    notes = [TK._convert_id_to_token(n) for n in out[0].detach().cpu().numpy()]
    notes = [note for note in notes if ('-' in note)]   #only keep the note tokens (skip condition tokens)
    midi_tok.remi_to_midi(notes, 'midi_recon.mid', ignore_velocity=False)