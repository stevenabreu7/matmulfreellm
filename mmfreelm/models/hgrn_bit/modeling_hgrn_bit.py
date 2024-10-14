# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
import typing as ty
from typing import List, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.utils import logging

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.quantization import (
    QuantizationConfig, OptionalFakeQuantize, swiglu_naive, FusedRMSNormSwishGateNaive
)
# from mmfreelm.layers.hgrn_bit import HGRNBitAttention  # NOTE: merged into this file!
from mmfreelm.models.utils import RecurrentCache
from mmfreelm.modules import FusedCrossEntropyLoss, RMSNorm, FusedRMSNormSwishGate, ShortConvolution
from mmfreelm.modules.activations import swiglu
from mmfreelm.ops.fusedbitnet import FusedBitLinear as FusedBitLinear
from mmfreelm.ops.fusedbitnet import BitLinear as UnfusedBitLinear
from mmfreelm.ops.fusedbitnet import RMSNormNaive
from mmfreelm.ops.hgrn.recurrent_fuse import fused_recurrent_hgrn
from mmfreelm.ops.hgrn.naive import naive_recurrent_hgrn


logger = logging.get_logger(__name__)


class HGRNBitAttention(nn.Module):
    """
    This implementation follows the HGRN2 paper:
    [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904)
    """
    def __init__(
        self,
        mode: str = 'fused_recurrent',
        hidden_size: int = 1024,
        num_heads: Optional[int] = None,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        layernorm_eps: float = 1e-5,
        layer_idx: int = None,
        quantization_cfg = None,
    ) -> HGRNAttention:
        super().__init__()

        # fused vs. naive
        self.use_unfused_recurrent = quantization_cfg.hgrnbitblock_config.attn_config.unfused_recurrent
        self.naive_rmsnormswish = quantization_cfg.hgrnbitblock_config.attn_config.naive_rmsnormswish
        self.naive_swiglu = quantization_cfg.hgrnbitblock_config.attn_config.naive_swiglu

        # activation quantization
        pow2scale = quantization_cfg.act_quant_pow2scale
        log_err = quantization_cfg.log_local_quant_errors
        quant = quantization_cfg.hgrnbitblock_config.attn_config.quant
        quant_rec_ht = quantization_cfg.hgrnbitblock_config.attn_config.quant_rec_ht
        # self.quant_it = OptionalFakeQuantize(quant, log_error=log_err)  # NOTE: c in paper -> i in code

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)
        self.head_dim = self.input_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel

        self.layer_idx = layer_idx

        assert mode in ['fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.hidden_size % num_heads == 0, f"hidden size must be divisible by num_heads of {num_heads}"

        if quantization_cfg.hgrnbitblock_config.attn_config.unfused_bitlinear:
            BitLinear = partial(UnfusedBitLinear, use_naive_norm=quantization_cfg.naive_rmsnorm)
        else:
            BitLinear = FusedBitLinear

        self.f_proj = BitLinear(hidden_size, self.input_dim, bias=False)
        self.quant_ft = OptionalFakeQuantize(quant, log_error=log_err, pow2scale=pow2scale)

        self.i_proj = BitLinear(hidden_size, self.input_dim, bias=False)
        self.quant_swiglu = OptionalFakeQuantize(quant, log_error=log_err, pow2scale=pow2scale)
        self.do_quant_rec_ht = quant_rec_ht is not None
        self.quant_rec_ht = OptionalFakeQuantize(quant_rec_ht, log_error=log_err, pow2scale=pow2scale)
        self.quant_ht = OptionalFakeQuantize(quant, log_error=log_err, pow2scale=pow2scale)

        assert not use_short_conv, "using short convolution"

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.f_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.i_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')

        self.g_proj = BitLinear(hidden_size, self.input_dim, bias=False)
        if self.naive_rmsnormswish:
            self.g_norm = FusedRMSNormSwishGateNaive(self.input_dim, eps=layernorm_eps, 
                                                     quant=quant, log_error=log_err)
        else:
            self.g_norm = FusedRMSNormSwishGate(self.input_dim, layernorm_eps)

        self.o_proj = BitLinear(self.input_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, (nn.Linear, FusedBitLinear, UnfusedBitLinear)):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.use_short_conv:
            # NOTE: quantization not supported here
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                i = self.i_proj(hidden_states)
                f = self.f_proj(hidden_states)
            else:
                conv_state_i = last_state[2] if use_cache else None
                conv_state_f = last_state[1] if use_cache else None
                i = self.i_conv1d(self.i_proj(hidden_states), attention_mask, conv_state_i)
                f = self.f_conv1d(self.f_proj(hidden_states), attention_mask, conv_state_f)
        else:
            i = self.i_proj(hidden_states)
            f = self.f_proj(hidden_states)

        f = f.sigmoid()
        f = self.quant_ft(f)  # NOTE: quantization

        # the lower bound for the first layer is zero
        if lower_bound is not None and self.layer_idx > 0:
            f = lower_bound + (1 - lower_bound) * f

        # swiglu activation
        if self.naive_swiglu:
            i = swiglu_naive(i, 1 - f)
        else:
            i = swiglu(i, 1 - f)
        i = self.quant_swiglu(i)  # NOTE: quantization

        # dealing with left-padding
        if attention_mask is not None:
            i = i.mul_(attention_mask.unsqueeze(-1))
        i, f = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (i, f))

        # recurrence
        recurrent_state = last_state[-1] if use_cache else None
        if mode == 'fused_recurrent' and not self.use_unfused_recurrent:
            o, recurrent_state = fused_recurrent_hgrn(i, f, initial_state=recurrent_state, output_final_state=use_cache)
        elif self.use_unfused_recurrent:
            if self.do_quant_rec_ht:
                dtype = i.dtype
                i = i.float()
                f = f.float()
                B, H, T, D = i.shape
                h = torch.zeros(B, H, D, dtype=torch.float, device=i.device)
                o = torch.zeros_like(i)
                final_state = None
                if recurrent_state is not None:
                    h += recurrent_state.detach()
                for t in range(T):
                    h = f[:, :, t] * h + i[:, :, t]
                    h = self.quant_rec_ht(h)  # NOTE: quantization
                    o[:, :, t] = h
                if use_cache:
                    final_state = h
                o = o.to(dtype)
                recurrent_state = final_state
                del h
                del final_state
                del dtype
            else:
                o, recurrent_state = naive_recurrent_hgrn(i, f, initial_state=recurrent_state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        o = self.quant_ht(o)  # NOTE: quantization

        if past_key_values is not None:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    last_state = (conv_state, recurrent_state)
                else:
                    last_state = (conv_state_i, conv_state_f, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, i.shape[2])

        o = self.g_norm(self.g_proj(hidden_states), rearrange(o, 'b h l d -> b l (h d)'))
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.hidden_size
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size


class HGRNBitMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        hidden_act: str = 'swish',
        intermediate_size: Optional[int] = None,
        quantization_cfg = None,
        layer_idx = None
    ) -> HGRNBitMLP:
        super().__init__()

        # setup for quantization & naive activation function
        pow2scale = quantization_cfg.act_quant_pow2scale
        log_err = quantization_cfg.log_local_quant_errors
        quant = quantization_cfg.hgrnbitblock_config.mlp_config.quant
        self.naive_swiglu = quantization_cfg.hgrnbitblock_config.mlp_config.naive_swiglu

        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        if quantization_cfg.unfused_bitlinear:
            BitLinear = partial(UnfusedBitLinear, use_naive_norm=quantization_cfg.naive_rmsnorm)
        else:
            BitLinear = FusedBitLinear

        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.quant_gate = OptionalFakeQuantize(quant, log_error=log_err, pow2scale=pow2scale)
        self.quant_swiglu = OptionalFakeQuantize(quant, log_error=log_err, pow2scale=pow2scale)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn = OptionalReLUify(fake_quant=fake_quant, act_fn=hidden_act)

    def forward(self, x):
        y = self.gate_proj(x)
        y = self.quant_gate(y)
        gate, y = y.chunk(2, -1)
        if self.naive_swiglu:
            swiglu_out = swiglu_naive(gate, y)
        else:
            swiglu_out = swiglu(gate, y)
        swiglu_out = self.quant_swiglu(swiglu_out)
        z = self.down_proj(swiglu_out)
        return z


class HGRNBitBlock(nn.Module):
    def __init__(self, config: HGRNBitConfig, layer_idx: int, quantization_cfg = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.remove_double_rmsnorms = quantization_cfg.hgrnbitblock_config.remove_double_rmsnorms
        pow2scale = quantization_cfg.act_quant_pow2scale
        log_err = quantization_cfg.log_local_quant_errors
        qconfig = quantization_cfg.hgrnbitblock_config
        self.naive_rms_norm = quantization_cfg.naive_rmsnorm
        rms_norm_cls = RMSNormNaive if self.naive_rms_norm else RMSNorm

        self.quant_input = OptionalFakeQuantize(qconfig.quant_input, log_error=log_err, pow2scale=pow2scale)
        self.attn_norm = rms_norm_cls(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.quant_postnorm1 = OptionalFakeQuantize(qconfig.quant_postnorm1, log_error=log_err, pow2scale=pow2scale)
        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
            quantization_cfg=quantization_cfg,
        )
        self.quant_attn = OptionalFakeQuantize(qconfig.quant_attn, log_error=log_err, pow2scale=pow2scale)
        self.mlp_norm = rms_norm_cls(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.quant_mlpnorm = OptionalFakeQuantize(qconfig.quant_mlpnorm, log_error=log_err, pow2scale=pow2scale)
        self.mlp = HGRNBitMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quantization_cfg=quantization_cfg,
            layer_idx=layer_idx,
        )
        self.quant_mlp = OptionalFakeQuantize(qconfig.quant_mlp, log_error=log_err, pow2scale=pow2scale)
        self.quant_residadd = OptionalFakeQuantize(qconfig.quant_residadd, log_error=log_err, pow2scale=pow2scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = self.quant_input(hidden_states)  # NOTE: quantization
        residual = hidden_states

        if not self.remove_double_rmsnorms:
            hidden_states = self.attn_norm(hidden_states)
            hidden_states = self.quant_postnorm1(hidden_states)  # NOTE: quantization

        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound
        )
        hidden_states = self.quant_attn(hidden_states)  # NOTE: quantization

        if not self.remove_double_rmsnorms:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
            hidden_states = self.quant_mlpnorm(hidden_states)  # NOTE: quantization

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.quant_mlp(hidden_states)  # NOTE: quantization

        hidden_states = residual + hidden_states
        hidden_states = self.quant_residadd(hidden_states)  # NOTE: quantization

        outputs = (hidden_states, attentions, past_key_values)

        return outputs


class HGRNBitPreTrainedModel(PreTrainedModel):

    config_class = HGRNBitConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['HGRNBitBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, FusedBitLinear, UnfusedBitLinear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class HGRNBitModel(HGRNBitPreTrainedModel):

    def __init__(self, config: HGRNBitConfig, quantization_cfg=None): 
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # NOTE(stevenabreu): for now we will ignore the embedding for quantization
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        assert quantization_cfg.quant_embedding is None, "Quantization of embedding is not supported yet."
        pow2scale = quantization_cfg.act_quant_pow2scale

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(torch.zeros(config.num_hidden_layers, config.hidden_size))
        self.layers = nn.ModuleList([
            HGRNBitBlock(config, layer_idx, quantization_cfg=quantization_cfg) 
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.remove_double_rmsnorm_final = quantization_cfg.remove_double_rmsnorm_final
        self.naive_rms_norm = quantization_cfg.naive_rmsnorm
        rms_norm_cls = RMSNormNaive if self.naive_rms_norm else RMSNorm

        self.norm = rms_norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.quant_norm = OptionalFakeQuantize(precision=quantization_cfg.quant_norm, log_error=quantization_cfg.log_local_quant_errors, pow2scale=pow2scale)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`HGRNBitModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None
            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    lower_bound
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=lower_bound
                )

            if output_attentions:
                all_attns += (attentions,)

        if not self.remove_double_rmsnorm_final:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.quant_norm(hidden_states)  # NOTE: quantization

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()
        if not return_dict:
            return tuple(x for x in [hidden_states, next_cache, all_hidden_states, all_attns] if x is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class HGRNBitForCausalLM(HGRNBitPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, quantization_cfg=None):
        super().__init__(config)

        if quantization_cfg is None:
            quantization_cfg = QuantizationConfig.NoneConfig()

        if quantization_cfg.unfused_bitlinear:
            BitLinear = partial(UnfusedBitLinear, use_naive_norm=quantization_cfg.naive_rmsnorm)
        else:
            BitLinear = FusedBitLinear

        self.model = HGRNBitModel(config, quantization_cfg=quantization_cfg)
        self.vocab_size = config.vocab_size
        # NOTE(stevenabreu): we can ignore the lm_head for quantization (it's bitlinear)
        assert quantization_cfg.quant_lm_head is None, "Quantization of LM head is not supported yet."
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values, input_ids.shape[1] - 1)
            input_ids, attention_mask = input_ids[:, -1:], attention_mask[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(logits.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
