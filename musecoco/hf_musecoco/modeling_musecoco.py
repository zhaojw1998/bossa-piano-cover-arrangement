# coding=utf-8
# Copyright 2024 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MuseCoco model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from typing import Dict, List, Optional
from .utils import read_json, jpath

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
# from ...activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_musecoco import MuseCocoConfig

from typing import Any, Optional

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "muzic/musecoco-large"
_CONFIG_FOR_DOC = "MuseCocoConfig"


# from ..deprecated._archive_maps import MUSECOCO_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.gpt2.modeling_gpt2.load_tf_weights_in_gpt2 with gpt2->musecoco
def load_tf_weights_in_musecoco(model, config, musecoco_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(musecoco_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2Attention with GPT2->MuseCoco
class MuseCocoAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `MuseCocoAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2FlashAttention2 with GPT2->MuseCoco
class MuseCocoFlashAttention2(MuseCocoAttention):
    """
    MuseCoco flash attention module. This module inherits from `MuseCocoAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        bsz, _, _ = hidden_states.size()
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `MuseCocoAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        query_length = query.shape[2]
        tgt_len = key.shape[2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query = query.transpose(1, 2).view(bsz, query_length, self.num_heads, self.head_dim)
        key = key.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.attn_dropout.p if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        if query.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.c_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query, key, value, attention_mask, query_length, dropout=attn_dropout
        )

        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        attn_output = self.c_proj(attn_weights_reshaped)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights_reshaped,)

        return outputs

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


MUSECOCO_ATTENTION_CLASSES = {
    "eager": MuseCocoAttention,
    "flash_attention_2": MuseCocoFlashAttention2,
}


class MuseCocoBlock(nn.Module):
    '''
    Move code from transformer_layer.py from MuseCoco
    '''
    def __init__(self, config, layer_idx=None):
        
        ''' -------- Start of Fairseq Code -------- '''
        super().__init__()
        self.embed_dim = config.n_embd
        self.dropout_module = nn.Dropout(config.dropout)

        self.quant_noise = 0 # getattr(config, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(config, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(config, "cross_self_attention", False)

        # Build the attention layer
        causal_linear_attention = CausalLinearAttention(self.embed_dim)
        linear_attention_layer = AttentionLayer(causal_linear_attention, 
                                                self.embed_dim, config.n_head, use_lora=config.use_lora)
        self.self_attn = linear_attention_layer

        self.activation_fn = nn.GELU()

        activation_dropout_p = 0
        self.activation_dropout_module = nn.Dropout(p=activation_dropout_p)

        self.normalize_before = config.norm_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(config, "char_inputs", False)
        self.self_attn_layer_norm = FairseqLayerNorm(self.embed_dim, export=export)

        self.encoder_attn = None
        self.encoder_attn_layer_norm = None

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)

        self.final_layer_norm = FairseqLayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        self.decoder_attention_heads = config.n_head
        ''' -------- End of Fairseq Code -------- '''


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        layer_past=None,
        use_cache=False,
    ):
        """
        The forward function copied from fairseq implementation.

        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # # attn is always a None value
        # x, attn = self.run_self_attn(   # This line produce nan in fp16 when x.max=12.3984
        #     query=x,
        #     key=y,
        #     value=y,
        #     key_padding_mask=self_attn_padding_mask,
        #     incremental_state=incremental_state,
        #     need_weights=False,
        #     attn_mask=self_attn_mask,
        # )

        ''' Do the self attention '''
        query = x.transpose(0, 1)
        key_padding_mask = self_attn_padding_mask
        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask
        # When query.max = 12.0391, below line output nan   12.3984
        r, present = self.self_attn(
            queries = query, 
            keys = query, 
            values = query,
            attn_mask=None, 
            key_padding_mask=key_padding_mask,
            layer_past=layer_past,
            use_cache=use_cache
        )  # Produce nan in fp16
        r = r.transpose(0, 1)
        x = r
        ''' Finish of self attention '''

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        assert self.encoder_attn is None and encoder_out is None

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            raise NotImplementedError
        
        attn = None
        return x, present, attn, None

    def run_self_attn(self, query, key_padding_mask, incremental_state, need_weights, **kwargs):
        if incremental_state is not None:
            raise NotImplementedError
        if need_weights:
            raise NotImplementedError

        query = query.transpose(0, 1)

        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask

        # When query.max = 12.0391, below line output nan   12.3984
        r = self.self_attn(query, query, query, attn_mask=None, key_padding_mask=key_padding_mask)  # Produce nan in fp16

        r = r.transpose(0, 1)

        return r, None
    
    def residual_connection(self, x, residual):
        '''
        Copied from fairseq.modules.transformer_layer.TransformerDecoderLayer.residual_connection()
        '''
        return residual + x


# Copied from transformers.models.gpt2.modeling_gpt2.GPT2PreTrainedModel with GPT2->MuseCoco,gpt2->musecoco,OpenAI GPT-2->MuseCoco
class MuseCocoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MuseCocoConfig
    load_tf_weights = load_tf_weights_in_musecoco
    base_model_prefix = "decoder"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["MuseCocoBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the MuseCoco Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


@dataclass
# Copied from transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput with GPT2->MuseCoco
class MuseCocoDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
            Multiple choice classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            MuseCocoAttentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


MUSECOCO_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MuseCocoConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MUSECOCO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the musecoco models have the
            following number of attention modules:

                - muzic/musecoco-large: 12
                - muzic/musecoco-large-medium: 24
                - muzic/musecoco-large-large: 36
                - muzic/musecoco-large-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using musecoco-xl, which has a total of 48 attention modules:
    model = MuseCocoLMHeadModel.from_pretrained("muzic/musecoco-large-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with muzic/musecoco-large-large:
    model = MuseCocoLMHeadModel.from_pretrained("muzic/musecoco-large-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


@add_start_docstrings(
    "The bare MUSECOCO Model transformer outputting raw hidden-states without any specific head on top.",
    MUSECOCO_START_DOCSTRING,
)
class MuseCocoModel(MuseCocoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)


        ''' -------- Code from fairseq -------- '''
        if getattr(config, "max_target_positions", None) is None:
            config.max_target_positions = config.n_positions # 8192 by default

        vocab_size = config.vocab_size
        embed_tokens = MuseCocoEmbedding(vocab_size, config.n_embd) # Use default padding id (1).

        self.config = config

        self._future_mask = torch.empty(0)

        self.dropout_module = nn.Dropout(config.dropout)

        self.decoder_layerdrop = config.decoder_layerdrop
        self.share_input_output_embed = False               # TODO: Maybe this config is not necessary 

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = config.n_embd
        self.embed_dim = embed_dim
        self.output_embed_dim = embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.n_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if not config.scale_emb else math.sqrt(embed_dim)

        self.quant_noise = None

        self.project_in_dim = None

        self.embed_positions = SinusoidalPositionalEmbedding(
            embedding_dim = config.n_embd,
            padding_idx = self.padding_idx,
            init_size = config.n_positions + self.padding_idx + 1,
        )

        if config.layernorm_emb:
            self.layernorm_embedding = FairseqLayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = config.cross_self_attention # False by default

        if self.decoder_layerdrop == 0:
            self.layers = nn.ModuleList([])
        else:
            raise NotImplementedError('Need to port LayerDropModuleList from fairseq')
        
        # Create Transformer layers
        no_encoder_attn = True
        self.layers.extend(
            [
                # self.build_decoder_layer(args, no_encoder_attn)
                # MuseCocoBlock(config, no_encoder_attn)
                MuseCocoBlock(config)
                for _ in range(config.n_layer)
            ]
        )
        self.num_layers = len(self.layers)

        if config.norm_before:
            self.layer_norm = FairseqLayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = None
        self.adaptive_softmax = None
        self.output_projection = None
        
        # self.output_projection = nn.Linear(
        #     self.output_embed_dim, config.vocab_size, bias=False
        # )
        # nn.init.normal_(
        #     self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
        # )

        self.gradient_checkpointing = getattr(self.config, 'gradient_checkpointing', False)
        if self.gradient_checkpointing:
            checkpointing_layers = getattr(self.config, 'gradient_checkpointing_layers', None)
            if checkpointing_layers is None:
                gradient_checkpointing_every_n_layer = getattr(self.config, 'gradient_checkpointing_every_n_layer', 1)
                checkpointing_layers = tuple(range(0, self.num_layers, gradient_checkpointing_every_n_layer))
            self.checkpointing_layers = checkpointing_layers
        ''' ----------------- '''


    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`MuseCocoModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.layers), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.layers))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.layers[block] = self.layers[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.layers)):
            self.layers[index] = self.layers[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.layers[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MUSECOCO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )

    def encode_input_ids(self, input_ids, past_length=0, position_ids=None, incremental_state=None, pos_start=0):
        device = input_ids.device
        input_shape = input_ids.size()
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Original Transformer's embed positions
        position_embeds = self.embed_positions(
                input_ids, incremental_state=incremental_state
            )

        if incremental_state is not None:
            input_ids = input_ids[:, -1:]
            if position_embeds is not None:
                position_embeds = position_embeds[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(input_ids)

        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if position_embeds is not None:
            if pos_start > 0:
                x[:, pos_start:] += position_embeds[:, :-pos_start]
            else:
                x += position_embeds

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x


    def forward(
            self,
            input_ids,
            past_key_values=None,
            use_cache=None,
            position_ids=None,
            sep_pos=None,
            encoder_out = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            output_hidden_states: Optional[bool] = None,
            inputs_embeds = None,
            self_attn_padding_mask = None,
            pos_start = 0
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        """
        docstring from extract feature function
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        '''
        According to fairseq_model.py:481, 
        prev_output_tokens = src_tokens 
        '''

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # Handle KV cache
        use_cache = use_cache if use_cache is not None else False
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_length = past_key_values[0][0].size(-2)

        if input_ids is not None:
            x = self.encode_input_ids(input_ids, past_length, position_ids, incremental_state, pos_start=pos_start)
        elif inputs_embeds is not None:
            x = inputs_embeds

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if self_attn_padding_mask is None:
            if self.cross_self_attention or input_ids.eq(self.padding_idx).any():
                self_attn_padding_mask = input_ids.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        # gradient_checkpointing_every_n_layer = getattr(self.args, "gradient_checkpointing_every_n_layer", 1)

        all_hidden_states = () if output_hidden_states else None

        # For KV cache
        presents = () if use_cache else None

        for idx, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            # if incremental_state is None and not full_context_alignment:
            #     self_attn_mask = self.buffered_future_mask(x)
            # else:
            #     self_attn_mask = None

            self_attn_mask = None  # Casual Linear Attention does not need this

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            if (
                getattr(self.config, "gradient_checkpointing", False) and self.training and
                idx in self.checkpointing_layers
            ):
                x, present, layer_attn, _ = checkpoint(
                    layer,
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    None,
                    None,
                    self_attn_mask,
                    self_attn_padding_mask,
                    bool((idx == alignment_layer)),
                    bool((idx == alignment_layer)),
                )
            else:
                x, present, layer_attn, _ = layer(  # This line produce nan when fp16 training, when x.max()=1061
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            # For KV cache
            if use_cache is True:
                presents = presents + (present,)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        extra = {"attn": [attn], "inner_states": inner_states}

        # # Apply LM head
        # if not features_only:
        #     x = self.output_layer(x)

        # Original GPT2's output format
        # ret = BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=presents,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )

        # Output format of original Fairseq model
        # ret = x, extra

        ret = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=x,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=None,
            cross_attentions=None,
        )

        return ret
    
    
@add_start_docstrings(
    """
    The MUSECOCO Model transformer with a language modeling head on top.
    Output layer are NOT tied to the input embeddings.
    """,
    MUSECOCO_START_DOCSTRING,
)
class MuseCocoLMHeadModel(MuseCocoPreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.decoder = MuseCocoModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`MuseCocoLMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.decoder.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.decoder.h))
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.decoder.deparallelize()
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[1]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    @add_start_docstrings_to_model_forward(MUSECOCO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_start=0
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        '''
        prev_output_tokens,
        sep_pos,
        encoder_out = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        '''

        # Original implementation
        transformer_outputs = self.decoder(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            sep_pos=None,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            full_context_alignment = False,
            alignment_layer = None,
            alignment_heads = None,
            src_lengths = None,
            return_all_hiddens = False,
            inputs_embeds=inputs_embeds,
            self_attn_padding_mask=attention_mask,
            pos_start=pos_start
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        ret = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

        return ret

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )



class MuseCocoPhraseClsModel8Head(nn.Module):
    '''
    The musecoco model with 2 classification head on top, for bar-level classification of 4-bar input phrase.
    '''
    def __init__(self, pt_ckpt, head1_vocab_size=13, head2_vocab_size=10, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        3 types + 1 NA for head2_vocab_size (quality)
        '''
        super().__init__()

        # Initialize from pretrained model (full LM)
        config_fp = jpath(pt_ckpt, 'config.json')
        config = read_json(config_fp)
        config = MuseCocoConfig.from_pretrained(config_fp)
        if random_init is False:
            musecoco_lm = MuseCocoLMHeadModel.from_pretrained(pt_ckpt, config=config)
        else:
            musecoco_lm = MuseCocoLMHeadModel(config)

        # Initialize MuseCocoModel 
        self.decoder = musecoco_lm.decoder

        self.cls_head_1_1 = nn.Linear(config.n_embd, head1_vocab_size)
        self.cls_head_2_1 = nn.Linear(config.n_embd, head1_vocab_size)
        self.cls_head_3_1 = nn.Linear(config.n_embd, head1_vocab_size)
        self.cls_head_4_1 = nn.Linear(config.n_embd, head1_vocab_size)

        self.cls_head_1_2 = nn.Linear(config.n_embd, head2_vocab_size)
        self.cls_head_2_2 = nn.Linear(config.n_embd, head2_vocab_size)
        self.cls_head_3_2 = nn.Linear(config.n_embd, head2_vocab_size)
        self.cls_head_4_2 = nn.Linear(config.n_embd, head2_vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        # Find the </s>'s position, the last token of the input
        s_positions = torch.where(input_ids[:,1:] == 2)[1] + 1

        # Original implementation
        transformer_outputs = self.decoder(
            input_ids,
            past_key_values=None,
            use_cache=False,
            sep_pos=None,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            full_context_alignment = False,
            alignment_layer = None,
            alignment_heads = None,
            src_lengths = None,
            return_all_hiddens = False,
        )
        hidden_states = transformer_outputs[0]  # [bs, seq_len, dim]

        # Get the output when </s> is input
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), s_positions, :] # [bs, dim]

        # Get outputs
        res1_1, res1_2 = self.cls_head_1_1(last_hidden_states), self.cls_head_1_2(last_hidden_states)
        res2_1, res2_2 = self.cls_head_2_1(last_hidden_states), self.cls_head_2_2(last_hidden_states)
        res3_1, res3_2 = self.cls_head_3_1(last_hidden_states), self.cls_head_3_2(last_hidden_states)
        res4_1, res4_2 = self.cls_head_4_1(last_hidden_states), self.cls_head_4_2(last_hidden_states)

        res1_out = torch.stack([res1_1, res2_1, res3_1, res4_1], dim=1) # [bs, pos, n_vocab1]
        res2_out = torch.stack([res1_2, res2_2, res3_2, res4_2], dim=1)

        return res1_out, res2_out


    

class MuseCocoPhraseClsModelDoubleHead(nn.Module):
    '''
    The musecoco model with 2 classification head on top, for bar-level classification of 4-bar input phrase.
    '''
    def __init__(self, pt_ckpt, head1_vocab_size=13, head2_vocab_size=10, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        3 types + 1 NA for head2_vocab_size (quality)
        '''
        super().__init__()

        # Initialize from pretrained model (full LM)
        config_fp = jpath(pt_ckpt, 'config.json')
        config = read_json(config_fp)
        config = MuseCocoConfig.from_pretrained(config_fp)
        if random_init is False:
            musecoco_lm = MuseCocoLMHeadModel.from_pretrained(pt_ckpt, config=config)
        else:
            musecoco_lm = MuseCocoLMHeadModel(config)

        # Initialize MuseCocoModel 
        self.decoder = musecoco_lm.decoder

        self.proj_pos1 = nn.Linear(config.n_embd, config.n_embd)
        self.proj_pos2 = nn.Linear(config.n_embd, config.n_embd)
        self.proj_pos3 = nn.Linear(config.n_embd, config.n_embd)
        self.proj_pos4 = nn.Linear(config.n_embd, config.n_embd)

        self.cls_head_1 = nn.Linear(config.n_embd, head1_vocab_size)
        self.cls_head_2 = nn.Linear(config.n_embd, head2_vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        # Find the </s>'s position, the last token of the input
        s_positions = torch.where(input_ids[:,1:] == 2)[1] + 1

        # Original implementation
        transformer_outputs = self.decoder(
            input_ids,
            past_key_values=None,
            use_cache=False,
            sep_pos=None,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            full_context_alignment = False,
            alignment_layer = None,
            alignment_heads = None,
            src_lengths = None,
            return_all_hiddens = False,
        )
        hidden_states = transformer_outputs[0]  # [bs, seq_len, dim]

        # Get the output when </s> is input
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), s_positions, :] # [bs, dim]

        h_pos1 = self.proj_pos1(last_hidden_states)
        h_pos2 = self.proj_pos2(last_hidden_states)
        h_pos3 = self.proj_pos3(last_hidden_states)
        h_pos4 = self.proj_pos4(last_hidden_states)

        res1_1, res1_2 = self.cls_head_1(h_pos1), self.cls_head_2(h_pos1) # res1_1: [bs, n_vocab1], res1_2: [bs, n_vocab2]
        res2_1, res2_2 = self.cls_head_1(h_pos2), self.cls_head_2(h_pos2)
        res3_1, res3_2 = self.cls_head_1(h_pos3), self.cls_head_2(h_pos3)
        res4_1, res4_2 = self.cls_head_1(h_pos4), self.cls_head_2(h_pos4)

        res1_out = torch.stack([res1_1, res2_1, res3_1, res4_1], dim=1) # [bs, pos, n_vocab1]
        res2_out = torch.stack([res1_2, res2_2, res3_2, res4_2], dim=1)

        return res1_out, res2_out


class MuseCocoPhraseClsModelSingleHead(nn.Module):
    '''
    The musecoco model with 2 classification head on top, for bar-level classification of 4-bar input phrase.
    '''
    def __init__(self, pt_ckpt, n_labels=13, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        '''
        super().__init__()

        # Initialize from pretrained model (full LM)
        config_fp = jpath(pt_ckpt, 'config.json')
        config = read_json(config_fp)
        config = MuseCocoConfig.from_pretrained(config_fp)
        if random_init is False:
            musecoco_lm = MuseCocoLMHeadModel.from_pretrained(pt_ckpt, config=config)
        else:
            musecoco_lm = MuseCocoLMHeadModel(config)

        # Initialize MuseCocoModel 
        self.decoder = musecoco_lm.decoder

        self.cls_head = nn.Linear(config.n_embd, n_labels)

        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        # Find the eos (</s>'s) position, the last token of the input
        s_positions = torch.where(input_ids[:,1:] == 2)[1] + 1

        # # Zero input baseline
        # input_ids = torch.zeros_like(input_ids)

        # Original implementation
        transformer_outputs = self.decoder(
            input_ids,
            past_key_values=None,
            use_cache=False,
            sep_pos=None,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            full_context_alignment = False,
            alignment_layer = None,
            alignment_heads = None,
            src_lengths = None,
            return_all_hiddens = False,
        )
        hidden_states = transformer_outputs[0]  # [bs, seq_len, dim]

        # Get the output when </s> is input
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), s_positions, :] # [bs, dim]

        res = self.cls_head(last_hidden_states)

        return res



@add_start_docstrings(
    """
    The MUSECOCO Model transformer with a sequence classification head on top (linear layer).

    [`MuseCocoForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MUSECOCO_START_DOCSTRING,
)
class MuseCocoForSequenceClassification(MuseCocoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.decoder = MuseCocoModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MUSECOCO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="microsoft/DialogRPT-updown",
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.decoder(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    MUSECOCO Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    MUSECOCO_START_DOCSTRING,
)
class MuseCocoForTokenClassification(MuseCocoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.decoder = MuseCocoModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MUSECOCO_INPUTS_DOCSTRING)
    # fmt: off
    @add_code_sample_docstrings(
        checkpoint="brad1141/musecoco-finetuned-comp2",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
        expected_output=[
            "Lead",
            "Lead",
            "Lead",
            "Position",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
            "Lead",
        ],
    )
    # fmt: on
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.decoder(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The GPT-2 Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MUSECOCO_START_DOCSTRING,
)
class MuseCocoForQuestionAnswering(MuseCocoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.decoder = MuseCocoModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MUSECOCO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def MuseCocoEmbedding(num_embeddings, embedding_dim, padding_id=1):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_id)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_id], 0)
    return m

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Copied from sinusoidal_positional_embedding.py of fairseq
    This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # bspair = torch.onnx.operators.shape_as_tensor(input)
        bspair = input.shape    # [06-15] Longshen: fix onnx.operator no attribute shape_as_tensor
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )
    

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 1.0

    def forward(self, x):
        return (self.lora_B @ self.lora_A @ x.permute(0, 2, 1)).permute(0, 2, 1) * self.scaling
    
class AttentionLayer(nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, use_lora=False):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.v_proj = nn.Linear(d_model, d_values * n_heads)
        self.out_proj = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.qkv_same_dim = d_keys == d_values == d_model // n_heads

        self.use_lora = use_lora
        if use_lora:
            self.lora_q = LoRALayer(d_model, d_keys * n_heads, rank=16)
            self.lora_k = LoRALayer(d_model, d_keys * n_heads, rank=16)
            self.lora_v = LoRALayer(d_model, d_keys * n_heads, rank=16)


    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
            self, 
            queries, 
            keys, 
            values, 
            attn_mask=None, 
            key_padding_mask=None,
            layer_past=None,
            use_cache=False,
        ):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # For KV cache. TODO: Maybe buggy
        if layer_past is not None:  # at some time, input (l=28) were concat with previous kv (l=27), strange
            past_key, past_value = layer_past
            keys = torch.cat((past_key, keys), dim=-2)
            values = torch.cat((past_value, values), dim=-2)
            pad_q = True
        else:
            pad_q = False
            
        if use_cache is True:
            present = (keys, values)
        else:
            present = None

        
        # Extract the dimensions into local variables
        N, query_seq_len, _ = queries.shape
        _, kv_seq_len, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        if not self.use_lora:
            queries = self.q_proj(queries).view(N, query_seq_len, H, -1)
            keys = self.k_proj(keys).view(N, kv_seq_len, H, -1)
            values = self.v_proj(values).view(N, kv_seq_len, H, -1)
        else:
            queries = (self.q_proj(queries) + self.lora_q(queries)).view(N, query_seq_len, H, -1)
            keys = (self.k_proj(keys) + self.lora_k(keys)).view(N, kv_seq_len, H, -1)
            values = (self.v_proj(values) + self.lora_v(values)).view(N, kv_seq_len, H, -1)

        # Compute the attention
        # Note: CausalLinearAttention has the assumption the Q, K, V has the same length
        # Need to pad the Q in the left side when inference with use_cache=True
        # with torch.autocast(device_type="cuda"):    when query = 54.75, below line produce nan
        attn_res = self.inner_attention(    # Produce nan in fp16   19:37 nan earlier than here
            queries,
            keys,
            values,
            attn_mask,
            key_padding_mask,
            pad_q=pad_q,
        ).reshape(N, query_seq_len, -1)   # original: view(N, L, -1)
        n_hidden = attn_res.shape[-1]

        ret = self.out_proj(attn_res) # Project the output and return

        return ret, present

    def pad_queries(self, queries, kv_seq_len):
        """
        Pad the queries tensor on the left side of the sequence length dimension.

        :param queries: Input tensor of shape [N, L, D]
        :param kv_seq_len: The target sequence length to pad to
        :return: Padded tensor of shape [N, kv_seq_len, D]
        """
        N, L, D = queries.shape
        if kv_seq_len <= L:
            return queries  # No padding needed if the target length is not greater than the current length

        # Calculate how much padding is needed
        pad_amount = kv_seq_len - L

        # Perform the padding (pad format: (padding_left, padding_right) for last dimension, and similar for others)
        # Since we only need to pad the second dimension (L), the padding configuration is as follows:
        # (0, 0) for the last dimension (D), (pad_amount, 0) for the second dimension (L), and (0, 0) for the first (N).
        padded_queries = F.pad(queries, (0, 0, pad_amount, 0), "constant", 0)

        return padded_queries

# from .fast_transformer.causal_product_python import causal_dot_product
from fast_transformers.causal_product import causal_dot_product

def causal_linear(Q, K, V):
    dtype = Q.dtype
    Q = Q.permute(0, 2, 1, 3).float().contiguous()  # [bs, n_head, seq_len, d_hidden]
    K = K.permute(0, 2, 1, 3).float().contiguous()
    V = V.permute(0, 2, 1, 3).float().contiguous()
    attn_res = causal_dot_product(Q, K, V)
    # This line produce number larger than 100k.
    # Cause overflow in the line below
    # V_new[V_new>65504] = 65504      # Added by Longshen, but problem persist
    # ret = torch.log()  # [bs, seq_len, n_head, d_hidden]
    ret = attn_res.permute(0, 2, 1, 3)
    ret = ret.type(dtype).contiguous()
    return ret 

class CausalLinearAttention(nn.Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.

    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """

    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):  # orig eps: 1e-6
        super(CausalLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps

    def forward(
            self, 
            queries, 
            keys, 
            values, 
            attn_mask=None, 
            key_padding_mask=None,
            pad_q=False,
        ):
        '''
        This layer produce nan in fp16 training.
        In debugging.
        :param queries:[B, L, H, D]
        :param keys:
        :param values:
        :param attn_mask:
        :param key_padding_mask:
        :return:
        '''

        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)  # [bs, seq_len (1), n_head, d_hidden]
        K = self.feature_map.forward_keys(keys) # [bs, prefix len, n_head, d_hidden]

        # Longshen: Padding the Q to be same length as K and V
        if pad_q:
            kv_seq_len = K.shape[1]
            pad_len = kv_seq_len - Q.shape[1]
            Q = self.pad_queries(Q, kv_seq_len)

        assert attn_mask is None, "Cannot assign attn_mask for %s" % self.__class__.__name__

        if key_padding_mask is not None:
            K = K * key_padding_mask.type(queries.dtype)[:, :, None, None]

        ''' Compute the normalizers and attention '''
            
        # Original code
        Z = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps) # Z.max() become inf in fp16   shape: [bs, pref_len, head]

        # # Longshen: einsum can produce inf in fp16.   (Q.max()=79)
        # t = torch.einsum("nlhi,nlhi->nlh", Q.float(), K.float().cumsum(1))
        # Z = 1 / (t + self.eps)  # in fp16, t can sometimes be 0, Z.max() become inf in fp16
        # clip_value = 1  # 1024
        # Z = torch.clamp(Z, min=-clip_value, max=clip_value)

        # Original code: Compute the unnormalized result
        V = causal_linear(  # shape: [bs, pref_len, n_head, dim]
            Q,
            K,
            values
        )           # This line produce inf
        ret = V * Z[:, :, :, None]  # This line produce nan in fp16    shape: [bs, pref_len, n_head, dim]

        # # Longshen: need to ensure the below operation is done in fp32
        # dtype = Q.dtype
        # V = causal_linear(
        #     Q.float(),
        #     K.float(),
        #     values.float(),
        # )

        # # Longshen: Then clip the attention score so that within range of fp16
        # ret = V * Z[:, :, :, None]  # This line produce nan in fp16
        # clip_value = 16
        # ret = torch.clamp(ret, min=-clip_value, max=clip_value)
        # ret = ret.type(dtype)  # convert back to fp16

        # Longshen: Unpad attn_res
        if pad_q is True:
            ret = ret[:, pad_len:,:,:]

        return ret
    
    def pad_queries(self, queries, kv_seq_len):
        """
        Pad the queries tensor on the left side of the sequence length dimension.

        :param queries: Input tensor of shape [bs, q_seq_len, n_head, d_hidden]
        :param kv_seq_len: The target sequence length to pad to
        :return: Padded tensor of shape [bs, kv_seq_len, n_head, d_hidden]
        """
        bs, q_seq_len, n_head, d_hidden = queries.shape
        if kv_seq_len <= q_seq_len:
            return queries  # No padding needed if the target length is not greater than the current length

        # Calculate how much padding is needed
        pad_amount = kv_seq_len - q_seq_len

        # Perform the padding
        # Since we only need to pad the second dimension (q_seq_len), the padding configuration is:
        # (0, 0) for the last dimension (d_hidden), 
        # (0, 0) for the third dimension (n_head),
        # (pad_amount, 0) for the second dimension (q_seq_len),
        # and (0, 0) for the first dimension (bs).
        padded_queries = F.pad(queries, (0, 0, 0, 0, pad_amount, 0, 0, 0), "constant", 0)

        return padded_queries

# from typing import Callable
# def get_activation_fn(activation: str) -> Callable:
#     """ Returns the activation function corresponding to `activation` """
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return gelu
#     elif activation == "gelu_fast":
#         deprecation_warning(
#             "--activation-fn=gelu_fast has been renamed to gelu_accurate"
#         )
#         return gelu_accurate
#     elif activation == "gelu_accurate":
#         return gelu_accurate
#     elif activation == "tanh":
#         return torch.tanh
#     elif activation == "linear":
#         return lambda x: x
#     else:
#         raise RuntimeError("--activation-fn {} not supported".format(activation))

def FairseqLayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def gelu(x: torch.Tensor) -> torch.Tensor:
    """GeLU activation function from Fairseq

    Args:
        x (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """    
    return torch.nn.functional.gelu(x.float()).type_as(x)

def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """
    Copied from fairseq.utils.make_positions

    Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

class FeatureMap(nn.Module):
    """
    Copied from fast_transformers.feature_maps.base.py

    Define the FeatureMap interface.
    """
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner

class ActivationFunctionFeatureMap(FeatureMap):
    """
    Copied from fast_transformers.feature_maps.base.py

    Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)

# Copied from fast_transformers.feature_maps.base.py
elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)

