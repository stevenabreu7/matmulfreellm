import math
import typing as ty
import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from transformers.activations import ACT2FN


@dataclass
class HGRNBitAttentionQuantizationConfig:
    quant: bool = False
    unfused_recurrent: bool = False
    naive_rmsnormswish: bool = False
    naive_swiglu: bool = False


@dataclass
class HGRNBitMLPQuantizationConfig:
    quant: bool = False
    naive_swiglu: bool = False


@dataclass
class HGRNBitBlockQuantizationConfig:
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
    log_local_quant_errors: bool = False
    unfused_bitlinear: bool = False
    naive_rmsnorm: bool = False

    @staticmethod
    def NoneConfig():
        return QuantizationConfig(
            hgrnbitblock_config = HGRNBitBlockQuantizationConfig(
                quant_input=None,
                quant_postnorm1=None,
                quant_attn=None,
                quant_mlpnorm=None,
                quant_mlp=None,
                quant_residadd=None,
                attn_config=HGRNBitAttentionQuantizationConfig(
                    quant=None,
                    unfused_recurrent=False,
                    naive_rmsnormswish=False,
                    naive_swiglu=False,
                ),
                mlp_config=HGRNBitMLPQuantizationConfig(
                    quant=None,
                    naive_swiglu=False,
                )
            ),
            quant_norm=None,
            quant_embedding=None,
            quant_lm_head=None,
            log_local_quant_errors=False,
            unfused_bitlinear=False,
            naive_rmsnorm=False,
        )


class CustomFakeQuantize(nn.Module):
    def __init__(self, 
                 precision=8,
                 qscheme=torch.per_tensor_symmetric, 
                 observer=MinMaxObserver,
                 log_error=True,
        ):
        super().__init__()

        assert precision <= 32, "precision must be <= 32"

        self.precision = precision
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
            self.zero_point = 0
        elif self.qscheme == torch.per_tensor_affine:
            self.scale = (max_val - min_val) / (self.quant_max - self.quant_min)
            self.zero_point = self.quant_min - torch.round(min_val / self.scale)
            self.zero_point = self.zero_point.clamp(self.quant_min, self.quant_max)
        else:
            raise ValueError(f'Unsupported quantization scheme: {self.qscheme}')

    def quantize_dequantize(self, x):
        # Quantize
        x_quantized = x / self.scale + self.zero_point
        if self.dtype == torch.qint8 or self.dtype == torch.quint8:
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
        self.maybe_quant = CustomFakeQuantize(**kwargs) if self.precision is not None else DummyQuantize()

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
