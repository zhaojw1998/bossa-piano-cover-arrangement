import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.cuda.amp import autocast as autocast

from transformers import MusicgenForConditionalGeneration, MusicgenConfig
from transformers.modeling_outputs import BaseModelOutput

from utils.lavis import all_gather_with_grad, concat_all_gather, BlipOutputOctMIDI, disabled_train
from utils.codebook_patterns import DelayedPatternProvider
from utils.inference import nucleus_filter

from model_Qformer import BertConfig, BertLMHeadModel
from musecoco.hf_musecoco.modeling_musecoco import MuseCocoLMHeadModel






class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class Blip2Base(nn.Module):
    """
    Adapted from LAVIS/BLIP-2/blip2.py. 
    Credit to https://github.com/salesforce/LAVIS
    """
    @classmethod
    def init_Qformer(cls, num_query_token, encoder_width, cross_attention_freq=2, load_pretrained=True, pretrained_path=None):
        # num_query_token: the number of query embeddings, set to 32.
        # encoder_width: the hidden size of the cross-attended encoder, set to the dmodel of MusicGen decoder.
        # cross_attention_freq: 2 means insert cross-attention every other layer.
        # load_pretrained: whether to load the pretrained weights of BERT-base model.
        # pretrained_path: path to the pretrained weights.

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")    #load BERT-base config from HuggingFace
        encoder_config.encoder_width = encoder_width
        
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        #specificed by MusicBERT
        encoder_config.vocab_size = 1237
        encoder_config.max_position_embeddings = 8194
        encoder_config.pad_token_id = 1

        Qformer = BertLMHeadModel(config=encoder_config)

        if load_pretrained:
            state_dict = torch.load("/data2/zhaojw/checkpoint_last_musicbert_base.pt", weights_only=False)['model']

            #rename the keys in MusicBERT weights to match BertLMHeadModel keys
            for key in list(state_dict.keys()):
                new_key = key.replace('encoder.sentence_encoder.layers', 'bert.encoder.layer')\
                            .replace('self_attn.k_proj', 'attention.self.key')\
                            .replace('self_attn.v_proj', 'attention.self.value')\
                            .replace('self_attn.q_proj', 'attention.self.query')\
                            .replace('self_attn.out_proj', 'attention.output.dense')\
                            .replace('self_attn_layer_norm', 'attention.output.LayerNorm')\
                            .replace('fc1', 'intermediate.dense')\
                            .replace('fc2', 'output.dense')\
                            .replace('final_layer_norm', 'output.LayerNorm')\
                            .replace('encoder.sentence_encoder.embed_tokens', 'bert.embeddings.word_embeddings')\
                            .replace('encoder.sentence_encoder.embed_positions', 'bert.embeddings.position_embeddings')\
                            .replace('encoder.sentence_encoder.emb_layer_norm', 'bert.embeddings.LayerNorm')\
                            .replace('encoder.lm_head.weight', 'cls.predictions.decoder.weight')\
                            .replace('encoder.lm_head.bias', 'cls.predictions.decoder.bias')\
                            .replace('encoder.lm_head.dense', 'cls.predictions.transform.dense')\
                            .replace('encoder.lm_head.layer_norm', 'cls.predictions.transform.LayerNorm')\
                            .replace('encoder.sentence_encoder.downsampling.0', 'bert.embeddings.downsampling')\
                            .replace('encoder.sentence_encoder.upsampling.0', 'upsampling')
                state_dict[new_key] = state_dict.pop(key)        

            Qformer.load_state_dict(state_dict, strict=False)
            for name, param in Qformer.named_parameters():
                if "_query" in name:
                    key_orig = name.replace("_query", "")
                    param.data.copy_(state_dict[key_orig])

        Qformer.resize_token_embeddings(1237+1) #add DEC token

        #init query tokens
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens


    def init_audio_encoder(self, musicgen_model="facebook/musicgen-large", load_pretrained=True):
        
        config=MusicgenConfig.from_pretrained(musicgen_model)
        #config.decoder.return_dict_in_generate = True
        #config.decoder.output_hidden_states = True
        num_layer = config.decoder.num_hidden_layers
        config.decoder.num_hidden_layers = num_layer//2 + 1   # take the hidden representation from the middle layer
        
        if load_pretrained:
            audio_encoder = MusicgenForConditionalGeneration.from_pretrained(musicgen_model, config=config)
        else:
            audio_encoder = MusicgenForConditionalGeneration(config=config)
        
        layer_norm = LayerNorm(config.decoder.hidden_size)

        return audio_encoder.decoder.model, audio_encoder.enc_to_dec_proj, audio_encoder.get_unconditional_inputs, layer_norm
    


class Blip2Qformer(Blip2Base):
    """
    First-stage model with Q-former (MusicBERT) and MusicGen.
    Adapted from LAVIS/BLIP-2/blip2_qformer.py. 
    Credit to https://github.com/salesforce/LAVIS
    """

    def __init__(self, num_query_token=32, cross_attention_freq=2, embed_dim=256, load_pretrained=True):
        super().__init__()

        #load musicgen model
        self.musicgen_decoder, \
        self.musicgen_enc_to_dec_proj, \
        self.musicgen_get_unconditional_inputs, \
        self.layer_norm = self.init_audio_encoder(musicgen_model="facebook/musicgen-large", load_pretrained=True)   #musicGEN
        #freeze musicgen model
        for _, param in self.musicgen_decoder.named_parameters():
            param.requires_grad = False
        self.musicgen_decoder = self.musicgen_decoder.eval()
        self.musicgen_decoder.train = disabled_train
        for _, param in self.musicgen_enc_to_dec_proj.named_parameters():
            param.requires_grad = False
        self.musicgen_enc_to_dec_proj = self.musicgen_enc_to_dec_proj.eval()
        self.musicgen_enc_to_dec_proj.train = disabled_train

        #load Q-former
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, \
                                                            self.musicgen_decoder.decoder.d_model, \
                                                            cross_attention_freq, \
                                                            load_pretrained)
        
        self.audio_proj = nn.Linear(self.Qformer.bert.config.hidden_size, embed_dim)
        self.symbo_proj = nn.Linear(self.Qformer.bert.config.hidden_size, embed_dim)

        self.binary_head = nn.Linear(self.Qformer.bert.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.patternprovider = DelayedPatternProvider(n_q=4)


    def forward(self, codec, notes, codec_pad_mask, leadsheet_mask, pad_mask):
        #codec: audio codec, (batch, 4, codec_len)
        #notes: symbolic tokens in OctMIDI encoding, (batch, 8*num_notes), including lead sheet (condition) + arrangement (output) notes
        #codec_pad_mask: (batch, codec_len+4); 1 for not masked, 0 for masked
        #leadsheet_mask: (batch, num_notes); differentiates leadsheet notes from the rest; 1 for masked, 0 for unmasked
        #notes_pad_mask: (batch, num_notes); 1 for not masked, 0 for masked
        
        bs = len(codec)

        with torch.no_grad():
            # get delayed codec pattern
            pattern = self.patternprovider.get_pattern(codec.shape[-1])
            gen_sequence, _, _ = pattern.build_pattern_sequence(codec, 2048)
            # get unconditional inputs for MusicGen
            unconditional_inputs = self.musicgen_get_unconditional_inputs(num_samples=bs)
            encoder_hidden_states = self.musicgen_enc_to_dec_proj(BaseModelOutput(*unconditional_inputs.encoder_outputs)[0].to(codec.device))
            encoder_attention_mask = unconditional_inputs.attention_mask.to(codec.device)
            # get audio hidden states
            audio_hidden_states = self.musicgen_decoder(
                                            input_ids=gen_sequence,
                                            encoder_attention_mask=encoder_attention_mask,
                                            encoder_hidden_states=encoder_hidden_states * encoder_attention_mask[..., None],
                                            return_dict=True,
                                        )[0]
        audio_embeds = self.layer_norm(audio_hidden_states)

        # expand query tokens to match batch size
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        # perform cross-attention and query the audio features
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=codec_pad_mask,
            use_cache=True,
            return_dict=True,
            is_decoder=False,
        )
        audio_feats = F.normalize(self.audio_proj(query_output.last_hidden_state), dim=-1)

        # get symbolic features
        symbo_output = self.Qformer.bert(notes,
                                   attention_mask=leadsheet_mask*pad_mask,
                                   is_decoder=False, 
                                   return_dict=True)
        symbo_feats = F.normalize(
            self.symbo_proj(symbo_output.last_hidden_state[torch.arange(bs, device=notes.device), 
                                                         torch.sum(leadsheet_mask==0, dim=-1, dtype=int),   # get the sos token's output (after the lead sheet condition)
                                                         :]), dim=-1
        )
                        
        ###============== Contrastive Loss ===================###
        audio_feats_all = all_gather_with_grad(audio_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        symbo_feats_all = all_gather_with_grad(symbo_feats)  # [batch_size*num_gpu, embed_dim]

        # audio-to-symbolic similarity
        sim_q2s = torch.matmul( # [bs, 1, num_query, embed_dim] x [bs*num_gpu, embed_dim, 1]
            audio_feats.unsqueeze(1), symbo_feats_all.unsqueeze(-1)
        ).squeeze() # [bs, bs*num_gpu, num_query_tokens]
        #aggregate across all query tokens
        sim_a2s, _ = sim_q2s.max(-1)
        sim_a2s = sim_a2s / self.temp

        # symbolic-to-audio symilarity: 
        sim_s2q = torch.matmul(   # [bs, 1, 1, embed_dim] x [bs*num_gpu, embed_dim, num_query_tokens]
            symbo_feats.unsqueeze(1).unsqueeze(1), audio_feats_all.permute(0, 2, 1)
        ).squeeze() #[bs, bs*num_gpu, num_query_tokens]
        # aggregate across all query tokens
        sim_s2a, _ = sim_s2q.max(-1)
        sim_s2a = sim_s2a / self.temp  # [batch_size, batch_size*num_gpu]

        # calculate loss
        rank = dist.get_rank()
        targets = torch.arange(start=rank*bs, end=(rank+1)*bs, step=1, dtype=int).to(codec.device)
        loss_contrastive = (
            F.cross_entropy(sim_a2s, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_s2a, targets, label_smoothing=0.1)
        ) / 2

        ###============== Matching Loss===================###
        symbo_input_ids_world = concat_all_gather(notes)
        symbo_attention_mask_world = concat_all_gather(leadsheet_mask*pad_mask)

        audio_embeds_world = all_gather_with_grad(audio_embeds)
        audio_attention_mask_world = all_gather_with_grad(codec_pad_mask)

        with torch.no_grad():
            sim_s2a[:, rank*bs : (rank+1)*bs].fill_diagonal_(-10000)
            sim_a2s[:, rank*bs : (rank+1)*bs].fill_diagonal_(-10000)            
                
            weights_s2a = F.softmax(sim_s2a, dim=1)
            weights_a2s = F.softmax(sim_a2s, dim=1)

        # select a negative audio for each symbolic
        audio_embeds_neg = []
        audio_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_s2a[b], 1).item()
            audio_embeds_neg.append(audio_embeds_world[neg_idx])
            audio_atts_neg.append(audio_attention_mask_world[neg_idx])
        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)
        audio_atts_neg = torch.stack(audio_atts_neg, dim=0)

        # select a negative symbolic for each audio
        symbo_input_ids_neg = []
        symbo_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_a2s[b], 1).item()
            symbo_input_ids_neg.append(symbo_input_ids_world[neg_idx])
            symbo_atts_neg.append(symbo_attention_mask_world[neg_idx])
        symbo_input_ids_neg = torch.stack(symbo_input_ids_neg, dim=0)
        symbo_atts_neg = torch.stack(symbo_atts_neg, dim=0)

        symbo_input_ids_all = torch.cat(
            [notes, notes, symbo_input_ids_neg], dim=0
        )  # pos, pos, neg
        symbo_atts_all = torch.cat(
            [leadsheet_mask*pad_mask, leadsheet_mask*pad_mask, symbo_atts_neg],
            dim=0,
        )

        query_tokens_matching = self.query_tokens.expand(symbo_input_ids_all.shape[0], -1, -1)
        query_atts_matching = torch.ones(query_tokens_matching.size()[:-1], dtype=torch.long).to(
            codec.device
        )
        attention_mask_all = torch.cat([query_atts_matching, symbo_atts_all], dim=1)

        audio_embeds_all = torch.cat(
            [audio_embeds, audio_embeds_neg, audio_embeds], dim=0
        )  # pos, neg, pos
        audio_atts_all = torch.cat(
            [codec_pad_mask, audio_atts_neg, codec_pad_mask],
            dim=0,
        )

        output = self.Qformer.bert(
            symbo_input_ids_all,
            query_embeds=query_tokens_matching,
            attention_mask=attention_mask_all,
            encoder_hidden_states=audio_embeds_all,
            encoder_attention_mask=audio_atts_all,
            return_dict=True,
            is_decoder=False,
        )

        vl_embeddings = output.last_hidden_state[:, : query_tokens_matching.size(1), :]
        vl_output = self.binary_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(codec.device)
        loss_matching = F.cross_entropy(logits, labels)

        ##================= language Model Loss ========================##
        input_ids = notes.clone()
        input_ids[input_ids==0] = 1237

        labels = input_ids.masked_fill(
            ~(leadsheet_mask*pad_mask).bool().repeat_interleave(8, dim=-1) | (input_ids==1237), -100
        )   #mask pad positions and special tokens

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(input_ids.device)
        attention_mask = torch.cat([query_atts, pad_mask], dim=1)
        
        lm_output = self.Qformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
            is_decoder=True,
            reduction="none",
        )
        
        loss_lm = lm_output.loss

        return BlipOutputOctMIDI(
            loss = loss_contrastive + loss_matching + loss_lm[0],
            loss_contrastive=loss_contrastive,
            loss_matching=loss_matching,
            loss_lm=loss_lm[0],
            loss_lm_bar=loss_lm[1],
            loss_lm_pos=loss_lm[2],
            loss_lm_ins=loss_lm[3],
            loss_lm_pch=loss_lm[4],
            loss_lm_dur=loss_lm[5],
            loss_lm_vel=loss_lm[6],
            loss_lm_ts=loss_lm[7],
            loss_lm_tmp=loss_lm[8],
        )


    @torch.no_grad()
    def test_contrastive(self, codec, notes, codec_pad_mask, leadsheet_mask, pad_mask):
        bs = len(codec)
        with torch.no_grad():
            pattern = self.patternprovider.get_pattern(codec.shape[-1])
            gen_sequence, _, _ = pattern.build_pattern_sequence(codec, 2048)
            unconditional_inputs = self.musicgen_get_unconditional_inputs(num_samples=bs)
            encoder_hidden_states = self.musicgen_enc_to_dec_proj(BaseModelOutput(*unconditional_inputs.encoder_outputs)[0].to(codec.device))
            encoder_attention_mask = unconditional_inputs.attention_mask.to(codec.device)

            audio_hidden_states = self.musicgen_decoder(
                input_ids=gen_sequence,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states * encoder_attention_mask[..., None],
                return_dict=True,
            )[0]
        audio_embeds = self.layer_norm(audio_hidden_states)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=codec_pad_mask,
            use_cache=True,
            return_dict=True,
            is_decoder=False,
        )

        audio_feats = F.normalize(
            self.audio_proj(query_output.last_hidden_state), dim=-1
        )
        
        symbo_output = self.Qformer.bert(notes, 
                                   attention_mask=leadsheet_mask*pad_mask, 
                                   is_decoder=False, 
                                   return_dict=True)
        
        symbo_feats = F.normalize(
            self.symbo_proj(symbo_output.last_hidden_state[torch.arange(bs, device=notes.device), 
                            torch.sum(leadsheet_mask==0, dim=-1, dtype=int), 
                            :]), dim=-1
        )

        ###============== Contrastive Loss ===================###
        audio_feats_all = all_gather_with_grad(audio_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        symbo_feats_all = all_gather_with_grad(symbo_feats)  # [batch_size*num_gpu, embed_dim]

        # audio-to-symbolic similarity
        sim_q2s = torch.matmul( # [bs, 1, num_query, embed_dim] x [bs*num_gpu, embed_dim, 1]
            audio_feats.unsqueeze(1), symbo_feats_all.unsqueeze(-1)
        ).squeeze() # [bs, bs*num_gpu, num_query_tokens]
        #aggregate across all query tokens
        sim_a2s, _ = sim_q2s.max(-1)
        
        target = torch.zeros(sim_a2s.shape, dtype=torch.long).to(codec.device)
        target.fill_diagonal_(1)
        
        loss_contrastive = F.cross_entropy(sim_a2s / self.temp, torch.arange(start=0, end=len(sim_a2s), step=1, dtype=int).to(codec.device), label_smoothing=0.1)

        sorted_indices = torch.argsort(sim_a2s, dim=-1, descending=True)
        # Sort the target tensor based on the sorted indices
        target = target.gather(-1, sorted_indices)
        non_zero_indices = torch.nonzero(target)

        return non_zero_indices, loss_contrastive


MODEL_SIZE = '1b'
PRETRAINED_MUSECOCO_PATH = '/data2/zhaojw/LAVIS/musecoco'

class Blip2Musecoco(nn.Module):
    """
    Second-stage model with Q-former and MuseCoco.
    Adapted from LAVIS/BLIP-2/blip2_t5.py. 
    Credit to https://github.com/salesforce/LAVIS
    """

    def __init__(self, load_pretrained=False, stage_1_checkpoint=None, musecoco_path=PRETRAINED_MUSECOCO_PATH, model_size=MODEL_SIZE):
        super().__init__()
        # load_pretrained: load pretrained MusicBERT and MusicGEN weights
        # stage_1_checkpoint: path to the checkpoint of the first-stage model

        self.blip2qformer = Blip2Qformer(load_pretrained=load_pretrained)
        
        if stage_1_checkpoint is not None:
            state_dict = torch.load(stage_1_checkpoint, map_location='cpu')
            for key in list(state_dict.keys()):
                new_key = key.replace('ln_audio', 'layer_norm')\
                            .replace('vision_proj', 'audio_proj')\
                            .replace('text_proj', 'symbo_proj')\
                            .replace('itm_head', 'binary_head')
                state_dict[new_key] = state_dict.pop(key)  
            self.blip2qformer.load_state_dict(state_dict)

        #freeze audio encoder
        for name, param in self.blip2qformer.musicgen_decoder.named_parameters():
            param.requires_grad = False
        self.blip2qformer.musicgen_decoder.eval()
        self.blip2qformer.musicgen_decoder.train = disabled_train
        for name, param in self.blip2qformer.musicgen_enc_to_dec_proj.named_parameters():
            param.requires_grad = False
        self.blip2qformer.musicgen_enc_to_dec_proj.eval()
        self.blip2qformer.musicgen_enc_to_dec_proj.train = disabled_train
        
        # delete the symbolic branch of Q-Former
        self.blip2qformer.Qformer.bert.embeddings.word_embeddings = None
        self.blip2qformer.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.blip2qformer.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.blip2qformer.Qformer.cls = None
        self.blip2qformer.binary_head = None
        self.blip2qformer.symbo_proj = None
        self.blip2qformer.audio_proj = None
        self.blip2qformer.Qformer.upsampling = None
        self.blip2qformer.Qformer.bert.embeddings.downsampling = None
        self.blip2qformer.temp = None

        #init musecoco model
        musecoco_fp = os.path.join(musecoco_path, model_size, 'model')
        self.musecoco = MuseCocoLMHeadModel.from_pretrained(musecoco_fp)

        # freeze musecoco except LoRA layers
        for name, param in self.musecoco.named_parameters():
            if not 'lora' in name:
                param.requires_grad = False

        self.musecoco_proj = nn.Linear(
            self.blip2qformer.Qformer.config.hidden_size, self.musecoco.config.n_embd
        )


    def forward(self, codec, token, codec_pad_mask, leadsheet_mask, pad_mask):
        # codec: audio codec, (batch, 4, codec_len)
        # token: symbolic tokens, (batch, token_seq_len), including lead sheet (condition) + arrangement (output) tokens
        # codec_pad_mask: (batch, codec_len+4); 1 for not masked, 0 for masked
        # leadsheet_mask, (batch, token_seq_len); False for not masked, True for masked
        # pad_mask, (batch, token_seq_len); False for not masked, True for masked

        with torch.no_grad():
            pattern = self.blip2qformer.patternprovider.get_pattern(codec.shape[-1])
            gen_sequence, _, _ = pattern.build_pattern_sequence(codec, 2048)
            unconditional_inputs = self.blip2qformer.musicgen_get_unconditional_inputs(num_samples=len(codec))
            encoder_hidden_states = self.blip2qformer.musicgen_enc_to_dec_proj(BaseModelOutput(*unconditional_inputs.encoder_outputs)[0].to(codec.device))
            encoder_attention_mask = unconditional_inputs.attention_mask.to(codec.device)

            audio_repr = self.blip2qformer.musicgen_decoder(
                input_ids=gen_sequence,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states * encoder_attention_mask[..., None],
                return_dict=True,
            )[0]

        audio_embeds = self.blip2qformer.layer_norm(audio_repr)
        query_tokens = self.blip2qformer.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_output = self.blip2qformer.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=codec_pad_mask,
            use_cache=False,
            return_dict=True,
            is_decoder=False,
        )

        inputs_musecoco = self.musecoco_proj(query_output.last_hidden_state)
        atts_musecoco = torch.zeros(query_tokens.size()[:-1], dtype=torch.long).bool().to(codec.device)

        targets = token.clone()
        targets = targets.masked_fill((leadsheet_mask | pad_mask), -100)

        inputs_embeds = self.musecoco.decoder.encode_input_ids(token, pos_start=61) # skip positional encoding for the first 61 tokens (attribute control tokens)
        inputs_embeds = torch.cat([inputs_musecoco, inputs_embeds], dim=1)
        decoder_atts = torch.cat([atts_musecoco, pad_mask], dim=1)

        targets = torch.nn.functional.pad(targets, (inputs_musecoco.size(1), 0), value=-100)

        outputs = self.musecoco(
            use_cache=True,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=decoder_atts,
            return_dict=True,
            labels=targets
        )

        loss = outputs.loss

        return {"loss": loss}


    def generate_nucleus2(self, codec, token, codec_pad_mask, leadsheet_mask=None, pad_mask=None, t=1, p=None, k=None):
        
        with torch.no_grad():
            pattern = self.blip2qformer.patternprovider.get_pattern(codec.shape[-1])
            gen_sequence, _, _ = pattern.build_pattern_sequence(codec, 2048)
            unconditional_inputs = self.blip2qformer.musicgen_get_unconditional_inputs(num_samples=len(codec))
            encoder_hidden_states = self.blip2qformer.musicgen_enc_to_dec_proj(BaseModelOutput(*unconditional_inputs.encoder_outputs)[0].to(codec.device))
            encoder_attention_mask = unconditional_inputs.attention_mask.to(codec.device)

            audio_repr = self.blip2qformer.musicgen_decoder(
                input_ids=gen_sequence,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states * encoder_attention_mask[..., None],
                return_dict=True,
            )[0]

            audio_embeds = self.blip2qformer.layer_norm(audio_repr)
            query_tokens = self.blip2qformer.query_tokens.expand(audio_embeds.shape[0], -1, -1)
            query_output = self.blip2qformer.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds, 
                encoder_attention_mask=codec_pad_mask,
                use_cache=False,
                return_dict=True,
                is_decoder=False,
            )

            inputs_musecoco = self.musecoco_proj(query_output.last_hidden_state)
            atts_musecoco = torch.zeros(query_tokens.size()[:-1], dtype=torch.long).bool().to(codec.device)
            
            if leadsheet_mask is not None:
                start_indices = torch.sum(leadsheet_mask.int(), dim=-1)[0]
                decoder_input_ids = token.clone()[:, :start_indices]
            else: 
                start_indices = 0
                decoder_input_ids = token.clone()

            from tqdm import tqdm
            count_bar = 0
            for _ in tqdm(range(2048 - decoder_input_ids.shape[1])):
                inputs_embeds = torch.cat([inputs_musecoco, self.musecoco.decoder.encode_input_ids(decoder_input_ids, pos_start=61)], dim=1)
                decoder_atts = torch.cat([atts_musecoco, torch.zeros(decoder_input_ids.shape).bool().to(codec.device)], dim=1)

                if _ == 0:
                    outputs = self.musecoco(
                        use_cache=True,
                        input_ids=None,
                        inputs_embeds=inputs_embeds,
                        attention_mask=decoder_atts,
                        return_dict=True
                    )
                else:
                    outputs = self.musecoco(
                        use_cache=True,
                        input_ids=None,
                        inputs_embeds=inputs_embeds[:, -1:],
                        attention_mask=decoder_atts,
                        return_dict=True,
                        past_key_values=outputs.past_key_values
                    )

                logits = outputs.logits[:, -1] 
                if p is not None:   #top-p sampling
                    logits = nucleus_filter(logits / t, p)
                elif (p is None) and (k is not None):   #top-k sampling
                    zeros = logits.new_ones(logits.shape) * float('-inf')
                    values, indices = torch.topk(logits, k, dim=-1)
                    logits = zeros.scatter(-1, indices, values/t)

                probability = torch.nn.functional.softmax(logits, dim=-1)
                prediction = torch.multinomial(probability, 1)
                if prediction[0] == 2:
                    break
                if prediction[0] == 5:
                    count_bar +=1
                if count_bar == 4:
                    #decoder_input_ids = torch.cat([decoder_input_ids, prediction], dim=-1)
                    break
                decoder_input_ids = torch.cat([decoder_input_ids, prediction], dim=-1)

        return decoder_input_ids[:, start_indices:]