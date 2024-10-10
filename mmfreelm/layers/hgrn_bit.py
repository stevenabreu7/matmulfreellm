# -*- coding: utf-8 -*-

# "HGRN2: Gated Linear RNNs with State Expansion"[https://arxiv.org/abs/2404.07904]

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from mmfreelm.modules import FusedRMSNormSwishGate, ShortConvolution
from mmfreelm.modules.activations import swiglu
from mmfreelm.ops.hgrn.recurrent_fuse import fused_recurrent_hgrn
from mmfreelm.ops.hgrn.naive import naive_recurrent_hgrn
from mmfreelm.models.hgrn_bit.quantization import OptionalFakeQuantize, swiglu_naive

#from mmfreelm.ops.bitnet import BitLinear_Fuse as BitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as BitLinear


# class FusedRMSNormSwishGateNaive(nn.Module):
#     def __init__(self, hidden_size, eps=1e-5, quant=None, log_error=False):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = None  # No bias in this implementation
#         self.quant_gnorm = OptionalFakeQuantize(quant, log_error=log_error)
#         self.quant_swish = OptionalFakeQuantize(quant, log_error=log_error)

#     def rms_norm(self, x):
#         # Root mean square normalization
#         rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
#         return x / rms * self.weight

#     def swish_gate(self, o):
#         # Swish gate: o * sigmoid(o)
#         return o * torch.sigmoid(o)

#     def forward(self, x, o, residual=None):
#         if residual is not None:
#             x = x + residual
#         # Apply RMS normalization to x
#         x_norm = self.rms_norm(x)
#         x_norm = self.quant_gnorm(x_norm)  # NOTE: quantization
#         # Apply Swish gate to o
#         gated_o = self.swish_gate(o)
#         gated_o = self.quant_swish(gated_o)  # NOTE: quantization
#         # Modulate normalized x with the gated o
#         return x_norm * gated_o


# class HGRNBitAttention(nn.Module):

#     def __init__(
#         self,
#         mode: str = 'fused_recurrent',
#         hidden_size: int = 1024,
#         num_heads: Optional[int] = None,
#         expand_ratio: Optional[int] = 1,
#         use_short_conv: bool = False,
#         conv_size: int = 4,
#         conv_bias: bool = False,
#         share_conv_kernel: bool = True,
#         layernorm_eps: float = 1e-5,
#         layer_idx: int = None,
#         quantization_cfg = None,
#     ) -> HGRNAttention:
#         super().__init__()

#         # fused vs. naive
#         self.use_unfused_recurrent = quantization_cfg.hgrnbitblock_config.attn_config.unfused_recurrent
#         self.naive_rmsnormswish = quantization_cfg.hgrnbitblock_config.attn_config.naive_rmsnormswish
#         self.naive_swiglu = quantization_cfg.hgrnbitblock_config.attn_config.naive_swiglu

#         # activation quantization
#         log_err = quantization_cfg.log_local_quant_errors
#         quant = quantization_cfg.hgrnbitblock_config.attn_config.quant
#         # self.quant_it = OptionalFakeQuantize(quant, log_error=log_err)  # NOTE: c in paper -> i in code

#         self.mode = mode
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.expand_ratio = expand_ratio
#         self.input_dim = int(hidden_size * expand_ratio)
#         self.head_dim = self.input_dim // self.num_heads

#         self.use_short_conv = use_short_conv
#         self.conv_size = conv_size
#         self.conv_bias = conv_bias
#         self.share_conv_kernel = share_conv_kernel

#         self.layer_idx = layer_idx

#         assert mode in ['fused_recurrent'], f"Not suppoerted mode `{mode}`."
#         assert self.hidden_size % num_heads == 0, f"hidden size must be divisible by num_heads of {num_heads}"

#         self.f_proj = BitLinear(hidden_size, self.input_dim, bias=False)
#         self.quant_ft = OptionalFakeQuantize(quant, log_error=log_err)

#         self.i_proj = BitLinear(hidden_size, self.input_dim, bias=False)
#         self.quant_swiglu = OptionalFakeQuantize(quant, log_error=log_err)
#         self.quant_ht = OptionalFakeQuantize(quant, log_error=log_err)

#         assert not use_short_conv, "using short convolution"

#         if use_short_conv:
#             self.conv_size = conv_size
#             if share_conv_kernel:
#                 self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
#             else:
#                 self.q_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
#                 self.f_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
#                 self.i_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')

#         self.g_proj = BitLinear(hidden_size, self.input_dim, bias=False)
#         if self.naive_rmsnormswish:
#             self.g_norm = FusedRMSNormSwishGateNaive(self.input_dim, eps=layernorm_eps, 
#                                                      quant=quant, log_error=log_err)
#         else:
#             self.g_norm = FusedRMSNormSwishGate(self.input_dim, layernorm_eps)

#         self.o_proj = BitLinear(self.input_dim, hidden_size, bias=False)

#         self.apply(self._initialize_weights)

#     def _initialize_weights(self, module):
#         if getattr(module, "_is_hf_initialized", False):
#             return
#         if isinstance(module, (nn.Linear, BitLinear)):
#             nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         module._is_hf_initialized = True

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[Cache] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#         lower_bound: Optional[torch.Tensor] = None,
#         **kwargs
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
#         # launching the triton kernel for just one token will actually be slower
#         mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

#         last_state = past_key_values[self.layer_idx] if use_cache else None
#         if self.use_short_conv:
#             # NOTE: quantization not supported here
#             conv_state = last_state[0] if use_cache else None
#             if self.share_conv_kernel:
#                 # conv state is updated inplace
#                 hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
#                 i = self.i_proj(hidden_states)
#                 f = self.f_proj(hidden_states)
#             else:
#                 conv_state_i = last_state[2] if use_cache else None
#                 conv_state_f = last_state[1] if use_cache else None
#                 i = self.i_conv1d(self.i_proj(hidden_states), attention_mask, conv_state_i)
#                 f = self.f_conv1d(self.f_proj(hidden_states), attention_mask, conv_state_f)
#         else:
#             i = self.i_proj(hidden_states)
#             f = self.f_proj(hidden_states)

#         f = f.sigmoid()
#         f = self.quant_ft(f)  # NOTE: quantization

#         # the lower bound for the first layer is zero
#         if lower_bound is not None and self.layer_idx > 0:
#             f = lower_bound + (1 - lower_bound) * f

#         # swiglu activation
#         if self.naive_swiglu:
#             i = swiglu_naive(i, 1 - f)
#         else:
#             i = swiglu(i, 1 - f)
#         i = self.quant_swiglu(i)  # NOTE: quantization

#         # dealing with left-padding
#         if attention_mask is not None:
#             i = i.mul_(attention_mask.unsqueeze(-1))
#         i, f = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (i, f))

#         # recurrence
#         recurrent_state = last_state[-1] if use_cache else None
#         if mode == 'fused_recurrent' and not self.use_unfused_recurrent:
#             o, recurrent_state = fused_recurrent_hgrn(i, f, initial_state=recurrent_state, output_final_state=use_cache)
#         elif self.use_unfused_recurrent:
#             o, recurrent_state = naive_recurrent_hgrn(i, f, initial_state=recurrent_state, output_final_state=use_cache)
#         else:
#             raise NotImplementedError(f"Not supported mode `{mode}`.")
#         o = self.quant_ht(o)  # NOTE: quantization

#         if past_key_values is not None:
#             if self.use_short_conv:
#                 if self.share_conv_kernel:
#                     last_state = (conv_state, recurrent_state)
#                 else:
#                     last_state = (conv_state_i, conv_state_f, recurrent_state)
#             else:
#                 last_state = (recurrent_state,)
#             past_key_values.update(last_state, self.layer_idx, i.shape[2])

#         o = self.g_norm(self.g_proj(hidden_states), rearrange(o, 'b h l d -> b l (h d)'))
#         o = self.o_proj(o)

#         return o, None, past_key_values

#     def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
#         param = next(self.parameters())
#         state = tuple()
#         if self.use_short_conv:
#             if self.share_conv_kernel:
#                 state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
#             else:
#                 state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),
#                           param.new_zeros(batch_size, self.hidden_size, self.conv_size),
#                           param.new_zeros(batch_size, self.hidden_size, self.conv_size))
#         state += (param.new_zeros(batch_size, self.num_heads, self.head_dim),)
#         return state

#     def state_size(self, **kwargs) -> int:
#         state_size = self.hidden_size
#         for module in self.children():
#             if isinstance(module, ShortConvolution):
#                 state_size += module.state_size
#         return state_size