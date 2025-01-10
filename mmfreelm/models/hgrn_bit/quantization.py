import math
import typing as ty
import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from transformers.activations import ACT2FN


@dataclass
class HGRNBitAttentionQuantizationConfig:
    quant: ty.Optional[int] = None
    quant_rec_ht: ty.Optional[int] = None
    unfused_recurrent: bool = False
    naive_rmsnormswish: bool = False
    naive_swiglu: bool = False
    unfused_bitlinear: bool = False


@dataclass
class HGRNBitMLPQuantizationConfig:
    quant: ty.Optional[int] = None
    naive_swiglu: bool = False


@dataclass
class HGRNBitBlockQuantizationConfig:
    remove_double_rmsnorms: bool = False
    quant_input: ty.Optional[int] = 8
    quant_postnorm1: ty.Optional[int] = 8
    quant_attn: ty.Optional[int] = 8
    quant_mlpnorm: ty.Optional[int] = 8
    quant_mlp: ty.Optional[int] = 8
    quant_residadd: ty.Optional[int] = 8
    attn_config: ty.Optional[HGRNBitAttentionQuantizationConfig] = HGRNBitAttentionQuantizationConfig()
    mlp_config: ty.Optional[HGRNBitMLPQuantizationConfig] = HGRNBitMLPQuantizationConfig()


@dataclass
class QuantizationConfig:
    hgrnbitblock_config: ty.Optional[HGRNBitBlockQuantizationConfig] = HGRNBitBlockQuantizationConfig()
    quant_norm: ty.Optional[int] = 8
    quant_embedding: ty.Optional[int] = None
    quant_lm_head: ty.Optional[int] = None
    remove_double_rmsnorm_final: bool = False
    log_local_quant_errors: bool = False
    unfused_bitlinear: bool = False
    naive_rmsnorm: bool = False
    act_quant_pow2scale: bool = False
    override_eps_1em3: bool = False
    override_eps_zero: bool = False
    rms_quant_act: ty.Optional[int] = None

    @staticmethod
    def NoneConfig():
        return QuantizationConfig(
            hgrnbitblock_config = HGRNBitBlockQuantizationConfig(
                remove_double_rmsnorms=False,
                quant_input=None,
                quant_postnorm1=None,
                quant_attn=None,
                quant_mlpnorm=None,
                quant_mlp=None,
                quant_residadd=None,
                attn_config=HGRNBitAttentionQuantizationConfig(
                    quant=None,
                    quant_rec_ht=None,
                    unfused_recurrent=False,
                    naive_rmsnormswish=False,
                    naive_swiglu=False,
                    unfused_bitlinear=False,
                ),
                mlp_config=HGRNBitMLPQuantizationConfig(
                    quant=None,
                    naive_swiglu=False,
                )
            ),
            quant_norm=None,
            quant_embedding=None,
            quant_lm_head=None,
            remove_double_rmsnorm_final=False,
            log_local_quant_errors=False,
            unfused_bitlinear=False,
            naive_rmsnorm=False,
            act_quant_pow2scale=False,
            override_eps_1em3=False,
            override_eps_zero=False,
            rms_quant_act=None,
        )
    
    @staticmethod
    def NaiveConfig():
        return QuantizationConfig(
            hgrnbitblock_config = HGRNBitBlockQuantizationConfig(
                remove_double_rmsnorms=False,
                quant_input=None,
                quant_postnorm1=None,
                quant_attn=None,
                quant_mlpnorm=None,
                quant_mlp=None,
                quant_residadd=None,
                attn_config=HGRNBitAttentionQuantizationConfig(
                    quant=None,
                    quant_rec_ht=None,
                    unfused_recurrent=True,  # naive
                    naive_rmsnormswish=True,  # naive
                    naive_swiglu=True,  # naive
                    unfused_bitlinear=True,  # naive
                ),
                mlp_config=HGRNBitMLPQuantizationConfig(
                    quant=None,
                    naive_swiglu=True,  # naive
                )
            ),
            quant_norm=None,
            quant_embedding=None,
            quant_lm_head=None,
            remove_double_rmsnorm_final=False,
            log_local_quant_errors=False,
            unfused_bitlinear=True,  # naive
            naive_rmsnorm=True,  # naive
            act_quant_pow2scale=False,
            override_eps_1em3=False,
            override_eps_zero=False,
            rms_quant_act=None,
        )


class CustomFakeQuantize(nn.Module):
    def __init__(self, 
                 precision,
                 qscheme=torch.per_tensor_symmetric, 
                 observer=MinMaxObserver,
                 log_error=False,
                 pow2scale=False,
        ):
        super().__init__()

        assert isinstance(precision, int), "precision must be an integer"
        assert 2 <= precision <= 32, "precision must be in [2, 32]"

        self.precision = precision
        self.pow2scale = pow2scale
        self.quant_min = -(2**(precision - 1))
        self.quant_max = 2**(precision - 1) - 1
        self.dtype = torch.qint8 if precision <= 8 else torch.qint32
        self.qscheme = qscheme

        # Initialize the observer
        self.activation_post_process = observer(dtype=self.dtype, qscheme=self.qscheme)
        self.scale = None
        self.zero_point = None

        # debugging information
        self.log_error = log_error
        if log_error:
            self.n_errors = 0
            self.sum_error = 0.0
            self.min_error = float('inf')
            self.max_error = 0.0
            self.mean_error = 0.0
            self.mean_error = 0
            self.m2 = 0  # helper for std
            # self.mean_error = []

    def forward(self, x):
        # Ensure the input is float32
        xdtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()
        
        # Update the observer with the current batch data
        self.activation_post_process(x)
        
        # Get quantization parameters from the observer
        self.calculate_qparams()
        
        # Apply fake quantization to simulate quantization during training
        x_pre = x.detach().cpu()
        x = self.quantize_dequantize(x)

        # Compute quantization errors and log them
        if self.log_error:
            err_abs = (x.detach().cpu() - x_pre).abs().mean().item()
            err_rel = err_abs / x_pre.abs().mean().item()
            self.sum_error += err_rel
            self.n_errors += 1
            self.min_error = min(self.min_error, err_rel)
            self.max_error = max(self.max_error, err_rel)
            # std
            delta = err_rel - self.mean_error
            self.mean_error += delta / self.n_errors
            delta2 = err_rel - self.mean_error
            self.m2 += delta * delta2

        return x.to(dtype=xdtype)
    
    def print_error_summary(self):
        if self.log_error:
            std = float('nan')
            if self.n_errors > 2:
                std = math.sqrt(self.m2 / (self.n_errors - 1))
            print(f"   Relative error: {self.sum_error / self.n_errors:6.2%} +- {std:6.2%}", end=" ")	
            print(f"(min: {self.min_error:6.2%}, max: {self.max_error:6.2%})")
            # mean_error = torch.tensor(self.mean_error)
            # print(f"   Relative error: {mean_error.mean().item():6.2%} +/- {mean_error.std().item():6.2%}", end=" ")
            # print(f"(min: {mean_error.min().item():6.2%}, max: {mean_error.max().item():6.2%})")
        else:
            print("log_error is turned off, cannot print error summary")

    def calculate_qparams(self):
        # Obtain min and max from the observer
        min_val, max_val = self.activation_post_process.min_val, self.activation_post_process.max_val

        # Compute scale and zero_point based on min and max
        if self.qscheme == torch.per_tensor_symmetric:
            max_abs = torch.max(-min_val, max_val)
            self.scale = max_abs / ((self.quant_max - self.quant_min) / 2)
            if self.pow2scale:
                self.scale = 2 ** torch.round(torch.log2(self.scale))
            self.zero_point = 0
        elif self.qscheme == torch.per_tensor_affine:
            self.scale = (max_val - min_val) / (self.quant_max - self.quant_min)
            if self.pow2scale:
                self.scale = 2 ** torch.round(torch.log2(self.scale))
            self.zero_point = self.quant_min - torch.round(min_val / self.scale)
            self.zero_point = self.zero_point.clamp(self.quant_min, self.quant_max)
        else:
            raise ValueError(f'Unsupported quantization scheme: {self.qscheme}')

    def quantize_dequantize(self, x):
        # Quantize
        x_quantized = x / self.scale + self.zero_point
        if self.dtype in (torch.qint8, torch.qint32):
            x_quantized = x_quantized.clamp(self.quant_min, self.quant_max)
        else:
            raise ValueError(f'Unsupported dtype: {self.dtype}')
        x_quantized = torch.round(x_quantized)
        
        # Dequantize
        x_dequantized = (x_quantized - self.zero_point) * self.scale
        return x_dequantized


class DummyQuantize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class OptionalFakeQuantize(nn.Module):
    def __init__(self, precision: ty.Optional[int] = None, **kwargs):
        super().__init__()
        self.precision = precision
        if self.precision is not None and isinstance(self.precision, int):
            self.maybe_quant = CustomFakeQuantize(precision=precision, **kwargs)
        else:
            self.maybe_quant = DummyQuantize()

    def forward(self, x):
        return self.maybe_quant(x)
    
    def print_error_summary(self):
        if self.precision is not None:
            self.maybe_quant.print_error_summary()


class OptionalReLUify(nn.Module):
    def __init__(self, fake_quant: bool = True, act_fn: str = "swish"):
        super().__init__()
        self.fake_quant = fake_quant
        self.act_fn = ACT2FN[act_fn]
        self.relu = nn.ReLU()
        self.mean_error = []

    def forward(self, x):
        if self.fake_quant:
            x_old = self.act_fn(x).detach().cpu()
            x_new = self.relu(x)
            err_rel = (x_new.detach().cpu() - x_old).abs().mean().item() / x_old.abs().mean().item()
            self.mean_error.append(err_rel)
            return x_new
        else:
            return self.act_fn(x)


def swiglu_naive(x, y):
    # silu_x = x / (1 + torch.exp(-x))
    # silu_x = x * torch.sigmoid(x)
    # return silu_x * y
    return x * torch.sigmoid(x) * y


class FusedRMSNormSwishGateNaive(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        quant=None,
        log_error=False,
        override_eps_zero=False,
        override_eps_1em3=False,
        quant_rms=None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = None  # No bias in this implementation
        self.quant_gnorm = OptionalFakeQuantize(quant, log_error=log_error)
        self.quant_swish = OptionalFakeQuantize(quant, log_error=log_error)

        self.quant_ms = OptionalFakeQuantize(quant_rms, log_error=log_error)
        self.quant_rms = OptionalFakeQuantize(quant_rms, log_error=log_error)
        self.quant_rms_inv = OptionalFakeQuantize(quant_rms, log_error=log_error)

        assert not (override_eps_zero and override_eps_1em3), "Cannot override both eps=0 and eps=1e-3"
        if override_eps_zero:
            self.eps = 0.0
        elif override_eps_1em3:
            self.eps = 1e-3

    def rms_norm(self, x):
        # Root mean square normalization
        ms = x.square().mean(dim=-1, keepdim=True) + self.eps
        ms = self.quant_ms(ms)
        if self.eps > 0.0:
            rms = torch.sqrt(ms)
            rms = self.quant_rms(rms)
            x_norm = x / rms
            x_norm = self.quant_rms_inv(rms)
        else:
            x_norm = x * 0.0
        return x_norm * self.weight

    def swish_gate(self, o):
        # Swish gate: o * sigmoid(o)
        return o * torch.sigmoid(o)

    def forward(self, x, o, residual=None):
        if residual is not None:
            x = x + residual
        # Apply RMS normalization to x
        x_norm = self.rms_norm(x)
        x_norm = self.quant_gnorm(x_norm)  # NOTE: quantization
        # Apply Swish gate to o
        gated_o = self.swish_gate(o)
        gated_o = self.quant_swish(gated_o)  # NOTE: quantization
        # Modulate normalized x with the gated o
        return x_norm * gated_o
