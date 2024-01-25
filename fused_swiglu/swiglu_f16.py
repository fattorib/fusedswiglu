"""Fused SwiGLU kernel in Triton"""
from typing import Tuple

import torch
from einops import rearrange
from torch.autograd import Function

from .kernels.kernels_fp16 import elementwise_backward, fused_bmm, fused_swiglu_fwd


def unbroadcast(x: torch.Tensor, dims: Tuple[int]):
    """Unbroadcast (sum) over dimensions."""
    return torch.sum(x, dim=dims, keepdim=True)


class FastAccumFusedSwiGLU(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        input: torch.HalfTensor,
        weight_gate: torch.HalfTensor,
        weight_up: torch.HalfTensor,
    ) -> torch.HalfTensor:
        out, x_weight_gate, x_weight_up = fused_swiglu_fwd(
            input, weight_gate, weight_up
        )
        ctx.save_for_backward(input, weight_gate, weight_up, x_weight_gate, x_weight_up)

        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input, weight_gate, weight_up, xw_gate, xw_up = ctx.saved_tensors

        # ---------------------------
        # Elementwise - Single Kernel
        # ---------------------------
        grad_xw_up, grad_xw_gate = elementwise_backward(grad_output, xw_gate, xw_up)

        # ---------------------------
        # Matmuls - Input Grad
        # ---------------------------
        grad_input = (grad_xw_gate @ weight_gate.T) + (grad_xw_up @ weight_up.T)

        # ---------------------------
        # Matmuls - Weight Grad
        # ---------------------------
        grad_weight_up, grad_weight_gate = fused_bmm(
            rearrange(input, "b sq d -> b d sq"), grad_xw_up, grad_xw_gate
        )

        return (
            grad_input,
            unbroadcast(grad_weight_gate, dims=(0,)),
            unbroadcast(grad_weight_up, dims=(0,)),
        )
