import math
import os

import numpy as np
import pytest
import torch

from .swiglu import FusedSwiGLU

torch.manual_seed(0)
np.random.seed(0)

try:
    NUM_TEST = int(os.environ["NUM_TEST"])
except Exception:
    NUM_TEST = 3

from typing import Tuple

BFLOAT16_MAX_TOL = 1e-2  # expect relative error for bf16 to be on order of 1e-3


def make_input_weights(
    bs: int, sq: int, d_model: int, d_intermediate: int
) -> Tuple[torch.HalfTensor, torch.HalfTensor, torch.HalfTensor]:
    input = torch.randn(
        (bs, sq, d_model), device="cuda:0", dtype=torch.bfloat16
    ).requires_grad_(True)

    w_gate = torch.randn(
        (d_model, d_intermediate), device="cuda:0", dtype=torch.bfloat16
    ) / (math.sqrt(d_model))
    w_up = torch.randn(
        (d_model, d_intermediate), device="cuda:0", dtype=torch.bfloat16
    ) / (math.sqrt(d_model))

    w_gate.requires_grad_(True)
    w_up.requires_grad_(True)

    return input, w_gate, w_up


def rel_error(x, y):
    return torch.linalg.norm(x - y) / torch.linalg.norm(y)


def swiglu_ref_torch(
    x: torch.HalfTensor, w_gate: torch.HalfTensor, w_up: torch.HalfTensor
) -> torch.HalfTensor:
    """Reference PyTorch implementation."""

    x_gate = torch.matmul(x, w_gate)
    x_up = torch.matmul(x, w_up)

    return torch.nn.functional.silu(x_gate) * x_up


@pytest.mark.parametrize(
    "dims",
    [
        (b, sq, d_model, d_intermediate)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for d_model in 64 * np.random.randint(1, 128, size=NUM_TEST)
        for d_intermediate in 64 * np.random.randint(1, 128, size=NUM_TEST)
    ],
)
def test_swiglu_fwd(dims):
    bs, sq, d_model, d_intermediate = dims

    input, w_gate, w_up = make_input_weights(bs, sq, d_model, d_intermediate)

    with torch.no_grad():
        torch_out = swiglu_ref_torch(input, w_gate, w_up)

        triton_out = FusedSwiGLU.apply(input, w_gate, w_up)

    assert rel_error(triton_out, torch_out) < BFLOAT16_MAX_TOL


@pytest.mark.parametrize(
    "dims",
    [
        (b, sq, d_model, d_intermediate)
        for b in np.random.randint(1, 4, size=NUM_TEST)
        for sq in 64 * np.random.randint(1, 32, size=NUM_TEST)
        for d_model in 64 * np.random.randint(1, 128, size=NUM_TEST)
        for d_intermediate in 64 * np.random.randint(1, 128, size=NUM_TEST)
    ],
)
def test_swiglu_bwd(dims):
    bs, sq, d_model, d_intermediate = dims

    input, w_gate, w_up = make_input_weights(bs, sq, d_model, d_intermediate)

    torch_out = swiglu_ref_torch(input, w_gate, w_up)

    dy = 0.1 * torch.randn_like(torch_out)

    torch_out.backward(dy, retain_graph=True)
    dinput_torch, dw_gate_torch, dw_up_torch = [
        _.grad.clone() for _ in [input, w_gate, w_up]
    ]
    input.grad, w_gate.grad, w_up.grad = None, None, None

    triton_out = FusedSwiGLU.apply(input, w_gate, w_up)
    triton_out.backward(dy, retain_graph=True)
    dinput_triton, dw_gate_triton, dw_up_triton = [
        _.grad.clone() for _ in [input, w_gate, w_up]
    ]

    assert rel_error(dinput_triton, dinput_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dw_gate_triton, dw_gate_torch) < BFLOAT16_MAX_TOL
    assert rel_error(dw_up_triton, dw_up_torch) < BFLOAT16_MAX_TOL
