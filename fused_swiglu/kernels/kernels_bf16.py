"""Kernel definitions for Forward/Backward kernels."""
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def silu(x):
    return (x * tl.sigmoid(x)).to(tl.bfloat16)


@triton.jit
def triton_silu_grad(x):
    return (tl.sigmoid(x) + (x * tl.sigmoid(x) * (1.0 - tl.sigmoid(x)))).to(tl.bfloat16)


# fmt: off
@triton.autotune(
        configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=["dim_m", "dim_n", "dim_k"],
)
@triton.jit
def swiglu_kernel(
    x_ptr,w_gate_ptr,w_up_ptr,o_ptr,act_w_gate_ptr,act_w_up_ptr,
    xbs_stride,xrow_stride,xcol_stride,
    wrow_stride,wcol_stride,
    obs_stride,orow_stride,ocol_stride,
    dim_m,dim_n,dim_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # fmt: on
    pid = tl.program_id(0)
    bs_pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_k, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_col
    group_id = pid // num_pid_in_group
    first_pid_row = group_id * GROUP_SIZE_M
    group_size_row = min(num_pid_row - first_pid_row, GROUP_SIZE_M)
    pid_row = first_pid_row + (pid % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc_gate = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    x_block_ptr = tl.make_block_ptr(
        x_ptr + bs_pid * xbs_stride,
        shape=(dim_m, dim_n),
        strides=(xrow_stride, xcol_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    w_gate_block_ptr = tl.make_block_ptr(
        w_gate_ptr,
        shape=(dim_n, dim_k),
        strides=(wrow_stride, wcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )

    w_up_block_ptr = tl.make_block_ptr(
        w_up_ptr,
        shape=(dim_n, dim_k),
        strides=(wrow_stride, wcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )

    for n in range(0, tl.cdiv(dim_n, BLOCK_SIZE_N)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1))

        w_gate_block = tl.load(w_gate_block_ptr, boundary_check=(0, 1))
        w_up_block = tl.load(w_up_block_ptr, boundary_check=(0, 1))

        acc_gate += tl.dot(
            x_block, w_gate_block, allow_tf32=False
        )
        acc_up += tl.dot(x_block, w_up_block, allow_tf32=False)


        x_block_ptr = tl.advance(x_block_ptr, offsets=(0, BLOCK_SIZE_N))
        w_gate_block_ptr = tl.advance(w_gate_block_ptr, offsets=(BLOCK_SIZE_N, 0))
        w_up_block_ptr = tl.advance(w_up_block_ptr, offsets=(BLOCK_SIZE_N, 0))


    act_w_gate_block_ptr = tl.make_block_ptr(
        act_w_gate_ptr + bs_pid * obs_stride,
        shape=(dim_m, dim_k),
        strides=(orow_stride, ocol_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    act_w_up_block_ptr = tl.make_block_ptr(
        act_w_up_ptr + bs_pid * obs_stride,
        shape=(dim_m, dim_k),
        strides=(orow_stride, ocol_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    tl.store(act_w_gate_block_ptr, acc_gate.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(act_w_up_block_ptr, acc_up.to(tl.bfloat16), boundary_check=(0, 1))

    acc_up *= silu(acc_gate)

    o_block_ptr = tl.make_block_ptr(
        o_ptr + bs_pid * obs_stride,
        shape=(dim_m, dim_k),
        strides=(orow_stride, ocol_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    tl.store(o_block_ptr, acc_up.to(tl.bfloat16), boundary_check=(0, 1))

def fused_swiglu_fwd(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    """
    Computes fused SwiGLU forward pass from `GLU Variants Improve Transformer`
    <https://arxiv.org/abs/2002.05202>
    """

    assert (
        x.shape[2] == w_up.shape[0]
    ), f"Dimension mismatch. Expected a.shape[2] ({x.shape[2]}) to be equal to b.shape[0] ({w_up.shape[0]})"

    assert (
        w_up.shape == w_gate.shape
    ), "Expected both weight matrices to have same shape"

    assert (
        x.ndim == 3 and w_up.ndim == 2
    ), "Incorrect number of dimensions for LHS or RHS"

    assert x.dtype == torch.bfloat16, "Expected torch.bfloat16 inputs and weights."

    B, M, N, K = x.shape[0], x.shape[1], x.shape[2], w_up.shape[1]
    out = torch.empty((B, M, K), device=x.device, dtype=x.dtype)

    # saved activations required for backward pass ((X @ W_G) & (X @ W_U))
    act_weight_gate = torch.empty_like(out)
    act_weight_up = torch.empty_like(out)
    
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
        B,
    )

    # fmt: off
    swiglu_kernel[grid](
        x,w_gate,w_up,out,
        act_weight_gate, act_weight_up,
        x.stride(0),x.stride(1),x.stride(2),
        w_up.stride(0),w_up.stride(1),
        out.stride(0),out.stride(1),out.stride(2),
        M,N,K,
    )
    # fmt: on
    return out, act_weight_gate, act_weight_up


# fmt: off
@triton.jit
def elementwise_backward_kernel(
    grad_out_ptr, xw_gate_ptr, xw_up_ptr,
    grad_xw_up_ptr,grad_xw_gate_ptr, 
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # fmt: on
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < (numel - (pid * BLOCK_SIZE))
    start_offs = pid*BLOCK_SIZE

    grad_out = tl.load(grad_out_ptr + start_offs + offs, mask = mask)
    xw_gate = tl.load(xw_gate_ptr + start_offs + offs, mask = mask)
    xw_up = tl.load(xw_up_ptr + start_offs + offs, mask = mask)

    xw_gate = xw_gate.to(tl.float32) 

    grad_xw_up = grad_out * silu(xw_gate)    
    grad_xw_gate = grad_out * xw_up * triton_silu_grad(xw_gate)

    tl.store(grad_xw_up_ptr + start_offs + offs, grad_xw_up.to(tl.bfloat16), mask = mask)
    tl.store(grad_xw_gate_ptr + start_offs + offs, grad_xw_gate.to(tl.bfloat16), mask = mask)

def elementwise_backward(grad_output: torch.HalfTensor, xw_gate: torch.HalfTensor, xw_up: torch.HalfTensor):
    """Fuses the elementwise backward operations to a single kernel."""

    grad_xw_up = torch.empty_like(grad_output)
    grad_xw_gate = torch.empty_like(grad_output)

    BLOCK_SIZE = 128
    num_warps = 4
    numel = grad_output.numel()

    grid = (numel // BLOCK_SIZE, )

    elementwise_backward_kernel[grid](
        grad_output, xw_gate, xw_up,
        grad_xw_up,grad_xw_gate,
        numel = numel, BLOCK_SIZE= BLOCK_SIZE,
        num_warps= num_warps
    )

    return grad_xw_up, grad_xw_gate


# fmt: off
@triton.autotune(
        configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=["dim_m", "dim_n", "dim_k"],
)
@triton.jit
def bmm_kernel(
    x_ptr,
    grad_xwup_ptr,grad_xwgate_ptr,
    grad_up_ptr,grad_gate_ptr,
    xbs_stride,xrow_stride,xcol_stride,
    grad_xwbs_stride,grad_xwrow_stride,grad_xwcol_stride,
    grad_bs_stride,grad_row_stride,grad_col_stride,
    dim_m,dim_n,dim_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # fmt: on
    pid = tl.program_id(0)
    bs_pid = tl.program_id(axis=1)

    num_pid_row = tl.cdiv(dim_m, BLOCK_SIZE_M)
    num_pid_col = tl.cdiv(dim_k, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_col
    group_id = pid // num_pid_in_group
    first_pid_row = group_id * GROUP_SIZE_M
    group_size_row = min(num_pid_row - first_pid_row, GROUP_SIZE_M)
    pid_row = first_pid_row + (pid % group_size_row)
    pid_col = (pid % num_pid_in_group) // group_size_row

    acc_up = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    acc_gate = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    x_block_ptr = tl.make_block_ptr(
        x_ptr + bs_pid * xbs_stride,
        shape=(dim_m, dim_n),
        strides=(xrow_stride, xcol_stride),
        offsets=(pid_row * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    xw_up_block_ptr = tl.make_block_ptr(
        grad_xwup_ptr + bs_pid * grad_xwbs_stride,
        shape=(dim_n, dim_k),
        strides=(grad_xwrow_stride, grad_xwcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )
    xw_gate_block_ptr = tl.make_block_ptr(
        grad_xwgate_ptr + bs_pid * grad_xwbs_stride,
        shape=(dim_n, dim_k),
        strides=(grad_xwrow_stride, grad_xwcol_stride),
        offsets=(
            0,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )

    for n in range(0, tl.cdiv(dim_n, BLOCK_SIZE_N)):
        x_block = tl.load(x_block_ptr, boundary_check=(0, 1))
        up_block = tl.load(xw_up_block_ptr, boundary_check=(0, 1))
        gate_block = tl.load(xw_gate_block_ptr, boundary_check=(0, 1))

        acc_up += tl.dot(x_block, up_block, allow_tf32=False)
        acc_gate += tl.dot(x_block, gate_block, allow_tf32=False)

        x_block_ptr = tl.advance(x_block_ptr, offsets=(0, BLOCK_SIZE_N))
        xw_up_block_ptr = tl.advance(xw_up_block_ptr, offsets=(BLOCK_SIZE_N, 0))
        xw_gate_block_ptr = tl.advance(xw_gate_block_ptr, offsets=(BLOCK_SIZE_N, 0))

    grad_up_block_ptr = tl.make_block_ptr(
        grad_up_ptr + bs_pid * grad_bs_stride,
        shape=(dim_m, dim_k),
        strides=(grad_row_stride, grad_col_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    acc_up = acc_up.to(tl.bfloat16)
    tl.store(grad_up_block_ptr, acc_up, boundary_check=(0, 1))

    grad_gate_block_ptr = tl.make_block_ptr(
        grad_gate_ptr + bs_pid * grad_bs_stride,
        shape=(dim_m, dim_k),
        strides=(grad_row_stride, grad_col_stride),
        offsets=(
            pid_row * BLOCK_SIZE_M,
            pid_col * BLOCK_SIZE_K,
        ),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )

    acc_gate = acc_gate.to(tl.bfloat16)
    tl.store(grad_gate_block_ptr, acc_gate, boundary_check=(0, 1))


def fused_bmm(x: torch.Tensor, grad_xw_up: torch.Tensor, grad_xw_gate: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    """Fuses the broadcasted w_gate and w_up matmuls to a single kernel."""

    assert (
        x.shape[-1] == grad_xw_up.shape[-2]
    ), f"Dimension mismatch. Expected a.shape[2] ({x.shape[-1]}) to be equal to b.shape[0] ({grad_xw_up.shape[-2]})"
    assert x.ndim == 3 and grad_xw_up.ndim == 3, "Incorrect number of dimensions for LHS or RHS"

    B, M, N, K = x.shape[0], x.shape[1], x.shape[2], grad_xw_up.shape[2]
    grad_weight_up = torch.empty((B, M, K), device=x.device, dtype=x.dtype)
    grad_weight_gate = torch.empty((B, M, K), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
        B,
    )

    #fmt: off
    bmm_kernel[grid](
        x,grad_xw_up, grad_xw_gate,
        grad_weight_up, grad_weight_gate,
        x.stride(0),x.stride(1),x.stride(2),
        grad_xw_up.stride(0),grad_xw_up.stride(1),grad_xw_up.stride(2),
        grad_weight_up.stride(0),grad_weight_up.stride(1),grad_weight_up.stride(2),
        M,N,K,
    )
    #fmt: on
    return grad_weight_up, grad_weight_gate
