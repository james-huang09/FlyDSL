# MonoKernel

`kernels.monokernel` is the shared home for resident, fused decoder-layer
kernels. Model ownership is explicit:

- `glm/` contains GLM-5 scheduling, indexed attention, wrappers, and golden
  references.
- `k3/` contains Kimi-K3 MLA/KDA state handling, AttnRes, latent-MoE logic,
  staged baselines, and benchmark tools.

Reusable contracts and primitives stay at this package root. `config.py`,
`layout.py`, `ops.py`, `packing.py`, `reference.py`, `runtime.py`, and
`weights.py` define shared geometry, layouts, device operations, packing, host
runtime, and weight containers. `gemm_a16w16.py`, `mxfp8_linear.py`, and
`symmetric_allreduce.py` provide model-independent kernels used by the staged
paths.

Keep model-specific schedules next to their model. In particular, Kimi-K3's
ordered KDA snapshot chain and MTP recurrence must remain in `k3/`; extracting
compiler-sensitive hot-loop code solely for API symmetry can change generated
GPU code and must be justified by performance measurements.

The resident-grid schedule, tagged-pair mailbox protocol, and phase-overlaid
LDS arena follow the TileRT/GLM MonoKernel design lineage. The comparison
reference used during this work is:

https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9

Model semantics and the Kimi-K3 ordered state pipeline are implemented in
FlyDSL and remain model-specific.

## GLM-5 MXFP4 layout and weight-address rotation

The GLM wrapper accepts the same row-major MXFP4 expert weights as before
(two E2M1 values per byte, one E8M0 scale per 32 values). It packs eight gate
rows followed by eight up rows into each 16-row MFMA tile, and tiles scales
by 16 rows and 128 K so each tile's 64 scale bytes are contiguous. Packing
happens once in the constructor. Activation quantization, BF16 MFMA arithmetic,
and down-projection prefetch batches are unchanged: up/gate uses FP8
activations and down uses BF16 activations.

Use the original functional and performance entry point:

```bash
python tests/kernels/test_glm5_monokernel.py --npes 8 -S 4 --pos 3000 --mxfp4
python tests/kernels/test_glm5_monokernel.py --npes 8 -S 4 --pos 3000 --mxfp4 --weight-copies 8
python tests/kernels/test_glm5_monokernel.py --npes 8 -S 4 --pos 3000 --mxfp4 --weight-copies 8 --bench --bench-iters 4096
```

`--weight-copies` defaults to 1; 8 allocates eight identical copies of every
final weight/scale/norm/bias tensor at distinct addresses. It works with the
FP8 path too, and may be combined with `--indexer`. Indexer pointer tables
are rebuilt for each copy. Scratch, peer buffers, epochs, and the compiled
launcher are shared, as for consecutive layers. Allocation and packing are
outside the measured region.

Weight sets are named `w_layer_0` through `w_layer_7` in the test code and
functional logs. `w_layer_0` uses the original tensors; the other sets hold
copies. Tensor names remain meaningful within each set, for example
`w_layer_1.w_ug` and `w_layer_1.s_ug`.

For every functional iteration, all copies run the same input through the
same stage checks, KV checks, and independent golden. The existing 5%
relative-L2 limit and routing-mismatch skip rule remain unchanged. Copies
must additionally produce bitwise-identical final outputs. Any check failure
makes the CLI exit nonzero. `-S` counts consecutive tokens of one request.

The benchmark cycles copy `layer % weight_copies` in the original HIP graph
(default 16 layers plus one step increment), with the original GPU-event
latency calculation. The layer count must be divisible by the copy count.
Compare `--weight-copies 1` and `8` with the same shape, input seed, graph
length, and iteration count; TP results also report the maximum rank latency.
These are synthetic identical-weight layers at different addresses, not
independently trained layers or eight independent requests. Benchmarking
alone does not run the functional acceptance checks.

Eight weight sets do not mean eight graph replays. A functional iteration
(`--iters 1`) calls each weight set once, for eight layer launches without a
HIP graph. A default benchmark graph has 16 layer launches, using
`w_layer_0` through `w_layer_7` twice. `--bench-iters 320` times 20 graph
replays; `--bench-iters 4096` times 256. These counts exclude warmup. The
benchmark prints the weight-set names, layers per graph, timed replays, and
timed layer launches before timing.
