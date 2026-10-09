# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GLM-5 indexed decode MonoKernel vs the torch golden.

Single GPU (TP1 view of one shard, the peer reduce is a 1-rank loopback)::

    python3 tests/kernels/test_glm5_monokernel.py --npes 1

TP8, one process per GPU::

    python3 tests/kernels/test_glm5_monokernel.py --npes 8

MXFP4 with eight weight addresses (same checks and benchmark as one address)::

    python3 tests/kernels/test_glm5_monokernel.py --npes 8 -S 4 --pos 3000 --mxfp4 --weight-copies 8
    python3 tests/kernels/test_glm5_monokernel.py --npes 8 -S 4 --pos 3000 --mxfp4 --weight-copies 8 --bench
"""

import argparse
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = str(get_rocm_arch() or "")
if _ARCH != "gfx950":
    pytest.skip(f"GLM-5 MonoKernel requires gfx950, got {_ARCH}", allow_module_level=True)

from kernels.monokernel.config import KV_LORA, PE_DIM  # noqa: E402
from kernels.monokernel.glm.layout import INDEX_DIM  # noqa: E402
from kernels.monokernel.glm.reference import golden_layer, golden_moe, indexer_golden, make_weights  # noqa: E402
from kernels.monokernel.reference import rope, rope_table  # noqa: E402

MAX_SEQ = 4096
TOL = {  # name -> (atol, rtol) on the fp32/bf16 intermediates
    "q_a": (2e-3, 2e-3),
    "kv_a": (2e-3, 2e-3),
    "q_nope": (5e-3, 5e-3),
    "q_pe": (5e-3, 5e-3),
    "q_lat": (1e-2, 1e-2),
    "o": (1e-2, 1e-2),
    "a": (3e-2, 2e-2),
    "scores": (1e-3, 1e-3),
    "prob": (1e-5, 1e-5),
    "kv": (2e-2, 1e-2),
    "x_out": (2e-2, 8e-3),  # BF16 peer partials can add ~2e-2 absolute error under cancellation
    "mid": (1e-3, 1e-3),  # FP8 x FP8 MFMA accumulation (~1e-4 abs), far below one E4M3 step
    "index_q": (2e-2, 2e-2),
    "index_w": (2e-3, 2e-3),
    "index_k": (2e-2, 2e-2),
}
# End to end, single FP8 rounding flips propagate, so judge by relative L2.
FP8_FLIPS = ("xq",)
REL_L2 = {
    "x_out_e2e": 5e-2,  # bf16 activation roundings at 5 MFMA inputs compound across ranks
}


def _check(name, got, ref, report):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    if name in FP8_FLIPS:
        # a 1-ulp f32 difference upstream may move a value across an FP8 rounding
        # midpoint: allow a few elements to differ by at most one E4M3 step (<= 1/8 rel)
        flips = err > 1e-5 * (1 + ref.abs())
        too_far = err > ref.abs() / 8 + 1e-4  # + subnormal step (block scale * 2^-9)
        report.append((name, err.max().item(), ref.abs().max().item(), f"flips={int(flips.sum())}"))
        return not bool(too_far.any()) and int(flips.sum()) <= max(4, ref.numel() // 1000)
    if name in REL_L2:
        rel = (err.norm() / ref.norm()).item()
        report.append((name, err.max().item(), ref.abs().max().item(), f"rel_l2={rel:.2e}"))
        return rel <= REL_L2[name]
    atol, rtol = TOL[name]
    bad = err > atol + rtol * ref.abs()
    report.append((name, err.max().item(), ref.abs().max().item(), f"bad={int(bad.sum())}"))
    return not bool(bad.any())


def _weight_layers(op, count):
    """Named weight sets at distinct addresses, sharing the native scratch/epochs/IPC.

    ``w_layer_0`` uses the original weights; ``w_layer_1`` through ``w_layer_7``
    hold copies of the final packed tensors. Each set retains tensor names such
    as ``w_ug`` and ``s_ug``, so a copy is identified as ``w_layer_1.w_ug``.
    Only ``op`` owns the peer buffer; borrowed layer ops must not be closed
    independently. Rebuild indexer pointer tables to use each set's addresses.
    """
    if count not in (1, 8):
        raise ValueError("weight_copies must be 1 or 8")
    base_weights = dict(op.W.t, **op.packed)
    layer_ops = {"w_layer_0": op}
    for layer_id in range(1, count):
        layer_name = f"w_layer_{layer_id}"
        layer_op = copy.copy(op)
        layer_op.W = copy.copy(op.W)
        layer_op.W.t = {tensor_name: tensor.clone() for tensor_name, tensor in base_weights.items()}
        layer_op.packed = {}
        if op.with_indexer:
            layer_op.index_params = torch.tensor(
                [
                    layer_op.W.t[name].data_ptr()
                    for name in (
                        "w_index_k",
                        "s_index_k",
                        "w_index_w",
                        "w_index_q",
                        "s_index_q",
                        "g_index_k",
                        "b_index_k",
                    )
                ]
                + [0 if op.timeline is None else op.timeline.data_ptr()],
                dtype=torch.int64,
                device=op.step.device,
            )
        layer_ops[layer_name] = layer_op
    for tensor_name in base_weights:
        pointers = {
            layer_name: dict(layer_op.W.t, **layer_op.packed)[tensor_name].data_ptr()
            for layer_name, layer_op in layer_ops.items()
        }
        assert len(set(pointers.values())) == count, f"weight layers alias for {tensor_name}: {pointers}"
    return layer_ops


def run_rank(
    rank,
    npes,
    S,
    cur_pos,
    iters,
    group=None,
    seed=1234,
    with_indexer=False,
    expert_mxfp4=False,
    weight_copies=1,
):
    from kernels.monokernel.glm import Glm5MonoKernel

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    topk = 2048
    W = make_weights(rank, heads=8, device=dev, seed=seed, with_indexer=with_indexer, expert_mxfp4=expert_mxfp4)
    cos, sin = rope_table(MAX_SEQ, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 99)  # same inputs on every rank
    kv0 = (torch.randn(MAX_SEQ, KV_LORA, generator=gen, device=dev)).to(torch.bfloat16)
    pe0 = (torch.randn(MAX_SEQ, PE_DIM, generator=gen, device=dev)).to(torch.bfloat16)
    if with_indexer:
        indices = torch.zeros(S, topk, dtype=torch.int32, device=dev)
        index0 = torch.randn(MAX_SEQ, INDEX_DIM, generator=gen, device=dev).to(torch.bfloat16)
    else:
        indices = torch.stack(
            [
                torch.randperm(max(cur_pos + s + 1, topk), generator=gen, device=dev)[:topk].sort().values
                for s in range(S)
            ]
        ).to(torch.int32)
        if cur_pos + 1 > topk:  # the reused selection always holds the newest token
            indices[:, -1] = torch.arange(cur_pos, cur_pos + S, device=dev, dtype=torch.int32)
        index0 = None
    op = Glm5MonoKernel(
        W,
        S,
        rank=rank,
        npes=npes,
        group=group,
        topk=topk,
        with_indexer=with_indexer,
        index_max_seq=MAX_SEQ,
    )
    layer_ops = _weight_layers(op, weight_copies)

    if npes == 1:
        allreduce = lambda x: x  # noqa: E731
    else:
        import torch.distributed as dist

        def allreduce(x):
            # The native peer protocol transmits BF16 partials, then sums in FP32.
            x = x.to(torch.bfloat16).float()
            parts = [torch.empty_like(x.cpu()) for _ in range(npes)]
            dist.all_gather(parts, x.cpu().contiguous(), group=group)
            return sum(parts[1:], parts[0]).to(x.device)

    ok = True
    for it in range(iters):
        h = (torch.randn(S, 6144, generator=gen, device=dev)).to(torch.bfloat16)
        for layer_name, layer_op in layer_ops.items():
            kv, pe = kv0.clone(), pe0.clone()
            index_cache = index0.clone() if with_indexer else None
            pos_t = torch.tensor([cur_pos], dtype=torch.int32, device=dev)
            out = layer_op.forward(h, pos_t, kv, pe, indices, cos, sin, index_cache=index_cache)
            torch.cuda.synchronize()
            got = layer_op.intermediates()
            got["x_out"] = out
            kv_ref, pe_ref = kv0.clone(), pe0.clone()
            report = []
            ref_indices = indices
            if with_indexer:
                index_ref = index0.clone()
                selected, index_q, index_w, _, _ = indexer_golden(
                    W, h, got["q_a"], cur_pos, index_ref, cos, sin, topk=topk
                )
                got_index_q = got["index_q"].clone()
                for s in range(S):
                    pos = cur_pos + s
                    got_index_q[s, :, :PE_DIM] = rope(got_index_q[s, :, :PE_DIM], cos[pos], sin[pos])
                got_index_q = got_index_q.to(torch.bfloat16).float()
                ok &= _check("index_q", got_index_q, index_q, report)
                ok &= _check("index_w", got["index_w"], index_w, report)
                rows = slice(cur_pos, cur_pos + S)
                ok &= _check("index_k", index_cache[rows], index_ref[rows], report)
                selected_ok = all(
                    torch.equal(got["indices"][s].sort().values, selected[s].sort().values) for s in range(S)
                )
                ok &= selected_ok
                report.append(("index_sel", 0.0, 0.0, f"set_equal={selected_ok}"))
                # Preserve the kernel's ascending-token output order so split-attention
                # rounding is compared independently of the selector's order.
                ref_indices = got["indices"]
            ref = golden_layer(W, h, cur_pos, kv_ref, pe_ref, ref_indices, cos, sin, allreduce, topk=topk)
            # attention half vs the full golden
            for name in ("q_a", "kv_a", "q_nope", "q_pe", "q_lat", "o", "a"):
                ok &= _check(name, got[name], ref[name], report)
            # MoE half vs the golden fed the kernel's own post-attention state, so a
            # 1-ulp difference in ``a`` cannot flip FP8 roundings downstream
            moe = golden_moe(W, got["a"].clone(), allreduce)
            for name in ("scores", "prob", "xq"):
                ok &= _check(name, got[name], moe[name], report)
            # up/gate + SiLU from the kernel's own FP8 activation
            ug = golden_moe(W, got["a"].clone(), allreduce, xq=got["xq"].clone())
            ok &= _check("mid", got["mid"], ug["mid"], report)
            # down + route weighting + TP reduce + residual from the kernel's own mid
            down = golden_moe(
                W,
                got["a"].clone(),
                allreduce,
                got["mid"].clone(),
                got["sel"].clone(),
                got["prob"].clone(),
            )
            ok &= _check("x_out", got["x_out"], down["x_out"], report)
            ok &= torch.equal(got["sel"], moe["sel"])
            # fully independent golden: only comparable when a 1-ulp difference in ``a``
            # did not flip a near-tied routing decision
            if torch.equal(ref["sel"], moe["sel"]):
                ok &= _check("x_out_e2e", got["x_out"], ref["x_out"], report)
            else:
                report.append(("x_out_e2e", 0.0, 0.0, "skipped: golden routing flipped on a 1-ulp difference in a"))
            rows = slice(cur_pos, cur_pos + S)
            ok &= _check("kv", kv[rows], kv_ref[rows], report)
            if rank == 0 or not ok:
                lines = [f"[rank {rank} iter {it} {layer_name}] ok={ok} sel_ok={torch.equal(got['sel'], moe['sel'])}"]
                lines += [f"   {n:9s} max_err={e:.3e} ref_max={m:.3e} {nb}" for n, e, m, nb in report]
                print("\n".join(lines), flush=True)
            if layer_name == "w_layer_0":
                first_output = out.clone()
            else:
                equal = torch.equal(out, first_output)
                ok &= equal
                if rank == 0 or not equal:
                    print(f"[rank {rank} iter {it} {layer_name}] bitwise_equal_to_w_layer_0={equal}", flush=True)
    op.close()
    return ok


def bench_rank(
    rank,
    npes,
    S,
    cur_pos,
    iters=320,
    group=None,
    seed=1234,
    with_indexer=False,
    expert_mxfp4=False,
    weight_copies=1,
):
    """HIP-graph replay of ``GLM5_LAUNCHES_PER_STEP`` layers; returns us per layer."""
    from kernels.monokernel.glm import Glm5MonoKernel

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    W = make_weights(rank, heads=8, device=dev, seed=seed, with_indexer=with_indexer, expert_mxfp4=expert_mxfp4)
    cos, sin = rope_table(MAX_SEQ, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 99)
    kv = torch.randn(MAX_SEQ, KV_LORA, generator=gen, device=dev).to(torch.bfloat16)
    pe = torch.randn(MAX_SEQ, PE_DIM, generator=gen, device=dev).to(torch.bfloat16)
    indices = torch.stack(
        [torch.randperm(max(cur_pos + s + 1, 2048), generator=gen, device=dev)[:2048] for s in range(S)]
    ).int()
    index_cache = (
        torch.randn(MAX_SEQ, INDEX_DIM, generator=gen, device=dev).to(torch.bfloat16) if with_indexer else None
    )
    launches_per_step = int(os.environ.get("GLM5_LAUNCHES_PER_STEP", "16"))
    if weight_copies not in (1, 8) or not 1 <= launches_per_step <= 128:
        raise ValueError("weight_copies must be 1 or 8 and GLM5_LAUNCHES_PER_STEP must be in [1, 128]")
    if launches_per_step % weight_copies or iters < launches_per_step:
        raise ValueError("benchmark needs complete weight rotations and at least one full graph replay")
    op = Glm5MonoKernel(
        W,
        S,
        rank=rank,
        npes=npes,
        group=group,
        launches_per_step=launches_per_step,
        with_indexer=with_indexer,
        index_max_seq=MAX_SEQ,
    )
    layer_ops = _weight_layers(op, weight_copies)
    weight_layer_names = tuple(layer_ops)
    h = torch.randn(S, 6144, generator=gen, device=dev).to(torch.bfloat16)
    x = torch.empty_like(h)
    pos_t = torch.tensor([cur_pos], dtype=torch.int32, device=dev)
    for warmup in range(max(10, weight_copies)):
        layer_name = weight_layer_names[warmup % weight_copies]
        layer_ops[layer_name].forward(h, pos_t, kv, pe, indices, cos, sin, x_out=x, index_cache=index_cache)
    torch.cuda.synchronize()
    import torch.distributed as dist

    if npes > 1:
        dist.barrier()
    if os.environ.get("MLA_MOE_TIMELINE"):
        top = Glm5MonoKernel(
            W,
            S,
            rank=rank,
            npes=npes,
            group=group,
            with_indexer=with_indexer,
            index_max_seq=MAX_SEQ,
            timeline=True,
        )
        for _ in range(3):
            top.forward(h, pos_t, kv, pe, indices, cos, sin, x_out=x, index_cache=index_cache)
        torch.cuda.synchronize()
        if rank == 0:
            print(top.timeline_report(), flush=True)
        top.close()
    # a HIP graph of LAYERS layer launches + one step bump, like a decode step
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(launches_per_step):
            layer_name = weight_layer_names[layer % weight_copies]
            layer_ops[layer_name].forward(
                h,
                pos_t,
                kv,
                pe,
                indices,
                cos,
                sin,
                x_out=x,
                layer=layer,
                advance=False,
                index_cache=index_cache,
            )
        op.advance_step()
    timed_replays = iters // launches_per_step
    if rank == 0:
        print(
            f"weight_layers={','.join(weight_layer_names)} graph_layers={launches_per_step} "
            f"timed_replays={timed_replays} timed_layer_launches={timed_replays * launches_per_step}",
            flush=True,
        )
    # Short decode graphs otherwise spend a material fraction of the timed
    # window ramping clocks from the idle state.  Scale warmup with the run so
    # isolated and already-busy-machine measurements remain comparable.
    for _ in range(max(3, min(50, iters // 64))):
        graph.replay()
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(timed_replays):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    elapsed = t0.elapsed_time(t1) * 1e3 / (timed_replays * launches_per_step)
    op.close()
    return elapsed


def _worker(rank, npes, S, cur_pos, iters, with_indexer, expert_mxfp4, weight_copies, results):
    import torch.distributed as dist

    master_port = os.environ.get("GLM5_MASTER_PORT", "29541")
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{master_port}", rank=rank, world_size=npes)
    if iters < 0:
        us = bench_rank(
            rank,
            npes,
            S,
            cur_pos,
            iters=-iters,
            with_indexer=with_indexer,
            expert_mxfp4=expert_mxfp4,
            weight_copies=weight_copies,
        )
        results[rank] = us
        print(f"[rank {rank}] {us:.1f} us/layer", flush=True)
    else:
        results[rank] = run_rank(
            rank,
            npes,
            S,
            cur_pos,
            iters,
            group=None,
            with_indexer=with_indexer,
            expert_mxfp4=expert_mxfp4,
            weight_copies=weight_copies,
        )
    dist.barrier()
    dist.destroy_process_group()


def run(npes, S, cur_pos, iters, with_indexer=False, expert_mxfp4=False, weight_copies=1):
    if npes == 1:
        return run_rank(
            0,
            1,
            S,
            cur_pos,
            iters,
            with_indexer=with_indexer,
            expert_mxfp4=expert_mxfp4,
            weight_copies=weight_copies,
        )
    import torch.multiprocessing as mp

    mgr = mp.Manager()
    results = mgr.dict()
    mp.spawn(
        _worker,
        args=(npes, S, cur_pos, iters, with_indexer, expert_mxfp4, weight_copies, results),
        nprocs=npes,
    )
    if iters < 0:
        print(f"max-rank {max(results.values()):.3f} us/layer", flush=True)
    return all(results[r] for r in range(npes))


@pytest.mark.parametrize("S,cur_pos", [(1, 100), (1, 3000), (2, 3000)])
def test_layer_single_gpu(S, cur_pos):
    assert run(1, S, cur_pos, 2)


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_layer_tp8():
    assert run(8, 1, 3000, 3)


@pytest.mark.parametrize("S", (1, 2, 4, 8))
def test_layer_mxfp4(S):
    assert run(1, S, 3000, 1, expert_mxfp4=True)


@pytest.mark.parametrize("expert_mxfp4,with_indexer", [(False, False), (True, False), (True, True)])
def test_layer_weight_rotation(expert_mxfp4, with_indexer):
    assert run(1, 4, 3000, 1, expert_mxfp4=expert_mxfp4, with_indexer=with_indexer, weight_copies=8)


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs")
def test_layer_mxfp4_tp8_weight_rotation():
    assert run(8, 4, 3000, 1, expert_mxfp4=True, weight_copies=8)


def test_mxfp4_scale_tiles():
    """Nonuniform scales must follow their 16-row tiles and K blocks exactly."""
    from kernels.monokernel.packing import pack_mxfp4_scale_tiles

    scales = torch.arange(2 * 32 * 8).remainder(11).add(117).to(torch.uint8).reshape(2, 32, 8)
    packed = pack_mxfp4_scale_tiles(scales).reshape(2, 2, 2, 16, 4)
    for e in range(2):
        for rg in range(2):
            for kc in range(2):
                for row in range(16):
                    assert torch.equal(packed[e, rg, kc, row], scales[e, rg * 16 + row, kc * 4 : kc * 4 + 4])


@pytest.mark.parametrize("S", (1, 4))
def test_layer_mxfp4_nonuniform_scales(S, monkeypatch):
    """Exercise scale addressing across rows, K blocks, experts, and paired gate/up."""
    original = make_weights

    def varied_weights(*args, **kwargs):
        weights = original(*args, **kwargs)
        gen = torch.Generator(device=weights.t["s_ug"].device).manual_seed(2026)
        for name, exponent in (("s_ug", 118), ("s_dn", 121)):
            scale = weights.t[name]
            scale.copy_(
                torch.randint(
                    exponent - 1, exponent + 2, scale.shape, generator=gen, device=scale.device, dtype=torch.uint8
                )
            )
        return weights

    monkeypatch.setitem(run_rank.__globals__, "make_weights", varied_weights)
    assert run(1, S, 3000, 1, expert_mxfp4=True, weight_copies=8)


@pytest.mark.parametrize(
    "S,cur_pos,expert_mxfp4",
    [(1, 0, False), (1, 2047, False), (1, 3000, False), (8, 0, True), (8, 3000, True)],
)
def test_layer_single_gpu_indexer(S, cur_pos, expert_mxfp4):
    assert run(1, S, cur_pos, 1, with_indexer=True, expert_mxfp4=expert_mxfp4)


def test_reference_default_dtype():
    """ATOM's BF16 default must not lower golden accumulators or probabilities."""
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        dev = torch.device("cuda", 0)
        W = make_weights(0, device=dev)
        gen = torch.Generator(device=dev).manual_seed(42)
        h = torch.randn(1, 6144, generator=gen, device=dev).bfloat16()
        kv = torch.randn(MAX_SEQ, KV_LORA, generator=gen, device=dev).bfloat16()
        pe = torch.randn(MAX_SEQ, PE_DIM, generator=gen, device=dev).bfloat16()
        indices = torch.arange(2048, dtype=torch.int32, device=dev)[None]
        cos, sin = rope_table(MAX_SEQ, device=dev)

        def allreduce(value):
            assert value.dtype == torch.float32
            return value

        outputs = []
        for dtype in (torch.float32, torch.bfloat16):
            torch.set_default_dtype(dtype)
            outputs.append(golden_layer(W, h, 100, kv.clone(), pe.clone(), indices, cos, sin, allreduce))
            assert outputs[-1]["prob"].dtype == torch.float32
        for name, value in outputs[0].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, outputs[1][name]), name
    finally:
        torch.set_default_dtype(original)


def test_router_preserves_close_scores():
    """A higher expert score must win even when its lead is below 256 FP32 ULPs."""
    from kernels.monokernel.glm import Glm5MonoKernel

    dev = torch.device("cuda", 0)
    W = make_weights(0, device=dev)
    W.t["w_r"].zero_()  # All uncorrected sigmoid scores are exactly 0.5.
    W.t["bias"].fill_(-10)
    W.t["bias"][:7] = 8  # Equal scores: lower expert IDs break the exact tie.
    W.t["bias"][92] = 7
    W.t["bias"][226] = 7 + 9.5367431640625e-5
    h = torch.ones(1, 6144, dtype=torch.bfloat16, device=dev)
    kv = torch.zeros(MAX_SEQ, KV_LORA, dtype=torch.bfloat16, device=dev)
    pe = torch.zeros(MAX_SEQ, PE_DIM, dtype=torch.bfloat16, device=dev)
    indices = torch.zeros(1, 2048, dtype=torch.int32, device=dev)
    cos, sin = rope_table(MAX_SEQ, device=dev)
    with Glm5MonoKernel(W, 1) as op:
        op.forward(h, torch.zeros(1, dtype=torch.int32, device=dev), kv, pe, indices, cos, sin)
        torch.cuda.synchronize()
        assert op.intermediates()["sel"].tolist() == [[256, 0, 1, 2, 3, 4, 5, 6, 226]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, default=1)
    ap.add_argument("-S", type=int, default=1)
    ap.add_argument("--pos", type=int, default=100)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--bench", action="store_true", help="time back-to-back launches")
    ap.add_argument("--bench-iters", type=int, default=320, help="layer launches timed by --bench")
    ap.add_argument("--indexer", action="store_true", help="fuse indexer projection, scoring, and top-k")
    ap.add_argument("--mxfp4", action="store_true", help="use native MXFP4 expert weights")
    ap.add_argument(
        "--weight-copies",
        type=int,
        choices=(1, 8),
        default=1,
        help="rotate identical packed weights at distinct addresses; check every copy with the same golden",
    )
    a = ap.parse_args()
    if a.iters < 1 or a.bench_iters < 1:
        ap.error("iteration counts must be positive")
    kwargs = dict(with_indexer=a.indexer, expert_mxfp4=a.mxfp4, weight_copies=a.weight_copies)
    if a.bench:
        if a.npes == 1:
            print(f"{bench_rank(0, 1, a.S, a.pos, iters=a.bench_iters, **kwargs):.1f} us/layer")
        else:
            run(a.npes, a.S, a.pos, -a.bench_iters, **kwargs)
        sys.exit(0)
    ok = run(a.npes, a.S, a.pos, a.iters, **kwargs)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
