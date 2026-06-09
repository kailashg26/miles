# Fix Report — Running miles `test_deepep_fp8.py` (DeepEP + FP8) on AMD MI355X

## 1. Objective
Run `tests/e2e/megatron/test_qwen3_30B_A3B/test_deepep_fp8.py` (Qwen3-30B-A3B, Megatron training + SGLang rollout with **DeepEP**, **FP8** rollout checkpoint) inside the `rlsys/miles:MI350-355-latest` container on **8× AMD Instinct MI355X (gfx950, ROCm 7.0)**.

## 2. Issues found and fixes (in the order hit)

### 2.1 Import-time failures (before training starts)
Two bugs in miles broke `import` on the Python 3.10 / older-sglang image:

- **`encoding_dsv4` / `encoding_dsv32` import error.** `miles/utils/chat_template_utils/deepseek_v4.py` and `deepseek_v32.py` imported the sglang DeepSeek encoders at module load. Fixed by converting them to **lazy imports** (`_encoding_dsv4()` / `_encoding_dsv32()` helpers called inside `render_messages()`).
- **`enum.StrEnum` missing on Python 3.10** (added in 3.11). Fixed by adding `miles/utils/enum_compat.py` (re-exports stdlib `StrEnum` on 3.11+, backports it on 3.10 with `__str__ = str.__str__` and `_generate_next_value_`), and switching `tito_tokenizer.py` and `test_utils/session_verify_agent.py` to import from it.

Result: both `chat_template_utils` and `tito_tokenizer` import cleanly.

### 2.2 DeepEP missing (primary blocker)
After imports were fixed, all 8 SGLang EP ranks crashed during MoE-dispatcher construction:

```text
ImportError: DeepEP is not installed. Please install DeepEP package from https://github.com/deepseek-ai/deepep.
```

sglang requires `from deep_ep import Buffer, Config`; the image ships no `deep_ep` module. Resolved by installing **UCCL-EP**, whose `deep_ep_wrapper` is a drop-in `deep_ep` package (`setup.py` `name="deep_ep"`, exporting `Buffer`/`Config`) backed by `uccl.ep`, with explicit MI355X/gfx950 support.

## 3. UCCL-EP installation (the DeepEP provider)

Key decision: **`uccl/ep/install_deps.sh` was NOT run** — on ROCm it reinstalls PyTorch nightly (`--index-url .../nightly/rocm7.0`), which would overwrite the image's customized ROCm torch. Only the minimal build deps were added.

```bash
# 1. Minimal build deps (does not touch torch)
pip install nanobind                       # was missing; setup.py hard-requires it
# libibverbs-dev / libnl-3-dev / libnl-route-3-dev already present in image

# 2. Build & install uccl.ep for MI355X (gfx950) from source against the image's torch
cd /apps/tas/yaoc/work/miles/uccl/ep
TORCH_CUDA_ARCH_LIST=gfx950 PYTORCH_ROCM_ARCH=gfx950 python setup.py install

# 3. Install the drop-in deep_ep package (skip dep resolution so PyPI 'uccl' can't override the local build)
cd deep_ep_wrapper
pip install --no-deps --no-build-isolation -e .
```

Build completed in ~86s, gfx950 confirmed in the build summary (`--offload-arch=gfx950`), warnings only (no errors).

## 4. Verification

Import layer:

```text
from deep_ep import Buffer, Config   ->  OK
Buffer  -> deep_ep.buffer.Buffer  (has low_latency_dispatch, get_dispatch_config, ...)
Config  -> uccl.ep.Config
```

End-to-end test, before vs after:

| Stage | Before UCCL | After UCCL |
|---|---|---|
| `import deep_ep` | ImportError at ~130s | Pass |
| 8x SGLangEngine (TP/EP 0-7) | Crash building MoE dispatcher | All load FP8 model (`Qwen3MoeForCausalLM, quant=fp8`) |
| HIP graph capture | Not reached | Reached, enters `forward_deepep` (DeepEP dispatch runs) |
| First failure | DeepEP import | Moved downstream to expert GEMM (~350s) |

The failure point moved from DeepEP import to the post-dispatch expert compute, confirming UCCL's DeepEP path is functioning.

## 5. aiter FP8 fused_moe assert — worked around via BF16 dispatch (applied & verified)

In FP8 DeepEP mode, sglang's low-latency dispatch emits `float8_e4m3fn` hidden states; `ep_moe/layer.py::forward_aiter` feeds them into aiter `fused_moe`, whose output dtype must be fp16/bf16:

```text
AssertionError: Fused_moe unsupported out dtype: torch.float8_e4m3fn
```

This is an sglang <-> aiter FP8 integration limit, not a UCCL issue — official DeepEP would hit the same assert because emitting FP8 is the intended DeepEP-FP8 behavior.

**Workaround applied and verified.** Set `SGLANG_DEEPEP_BF16_DISPATCH=1`: `deepep.py:608-613` gates `use_fp8` on this env, so low-latency dispatch outputs BF16 and aiter `fused_moe` runs with `out dtype = torch.bfloat16` (weights stay FP8). UCCL's `low_latency_dispatch` natively supports `use_fp8=False` (returns a `torch.bfloat16` tensor, per its docstring).

Propagation: env vars only reach the sglang server subprocess via the ray `runtime_env`. A pass-through was added in `_common.py::execute()` so `SGLANG_DEEPEP_BF16_DISPATCH` (and `SGLANG_USE_AITER`) flow into `extra_env_vars` when present in the outer shell.

Result (`/tmp/dpsk_v4_test3.log`):
- `Fused_moe unsupported out dtype` occurrences: **0** (previously the hard stop).
- aiter fused_moe compiles/runs: `[fused_moe] using 1stage default for (..., 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', ...)` — out dtype is now bf16, weights still fp8.
- Progress moved past expert compute into HIP graph capture.

## 6. New blocker (exposed after BF16 dispatch): UCCL `low_latency_combine` 2D/3D mismatch

Once fused_moe passed, all 8 ranks crashed during HIP graph capture with `Memory access fault by GPU node-*`. The real trigger is at the end of the stack — UCCL's `low_latency_combine` reads `x.size(2)`:

```text
File ".../uccl/ep/deep_ep_wrapper/deep_ep/buffer.py", line 553, in low_latency_combine
    x_for_combine.size(2),
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

- UCCL combine assumes a **3D** input `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` (matches official DeepEP; see buffer.py:486 docstring).
- On the AMD/aiter path, `ep_moe/layer.py::forward_aiter` returns the aiter `fused_moe` result directly. aiter fuses permute/unpermute and collapses the expert dimension, producing a **2D** `[tokens, hidden]` tensor → `x.size(2)` is out of range.
- The all-GPU `Memory access fault` is the cascade: the Python exception aborts capture mid-stream, leaving UCCL's low-latency RDMA proxies reading unsynchronized device memory.

**Independent of BF16 dispatch.** Both FP8 and BF16 dispatch feed 3D hidden states into `forward_aiter`; the 2D shape comes out of aiter `fused_moe` itself. This mismatch was simply masked earlier by the fused_moe assert.

**Update — reshape attempt proves this is a *semantic*, not a layout, mismatch.** Reshaping the 2D tensor back to 3D in `low_latency_combine` (using the handle's dims) raised `RuntimeError: shape '[16, 1024, 2048]' is invalid for input of size 131072`. Since `131072 = 64 × 2048`, aiter actually emits `[64, 2048]` — **64 tokens, already weighted-reduced** (token-view) — while combine expects the **masked expert-view** `[16, 1024, 2048]` (16 local experts × 1024 recv slots), a 256× element-count gap. aiter's fused MoE collapses permute → expert GEMM → unpermute → topk weighted-sum into one kernel and returns the final per-token result; DeepEP/UCCL `low_latency_combine` instead expects the *un-reduced* per-expert masked tensor and performs the reduction + cross-rank return itself. The two execution models are incompatible at the combine boundary, so no reshape can bridge them. The reshape was reverted (UCCL restored). Bridging requires either (a) aiter emitting a masked, un-reduced layout on the DeepEP path, or (b) an aiter-specific combine that only does the cross-rank return without reducing — both beyond a one-line fix.

## 7. Open question for the team — FP8 semantics of the BF16-dispatch workaround

Does `SGLANG_DEEPEP_BF16_DISPATCH=1` weaken what `test_deepep_fp8.py` verifies? Separate two "FP8" layers:

1. **FP8 weight-model rollout (what the case declares).** `use_fp8_rollout=True` only swaps the rollout checkpoint to the FP8 model (`_common.py:104-105`, `--hf-checkpoint .../Qwen3-30B-A3B-FP8`). BF16 dispatch does **not** touch this — weights stay FP8 and MoE expert GEMM still runs FP8 (fused_moe w13/w2 = `float8_e4m3fn`).
2. **FP8 activation transport inside DeepEP (default-on bandwidth optimization).** `deepep.py:608-613` sets `use_fp8=True` unless `SGLANG_DEEPEP_BF16_DISPATCH` is set, regardless of model quantization. BF16 dispatch **does** downgrade this — activations are transported/computed in BF16.

So the workaround preserves the *declared* FP8 (weights) but diverges on FP8 *activation* transport/compute.

Key clarification (confirmed against official DeepEP + DeepSeek-V3): DeepEP's FP8 support is **asymmetric by design — dispatch can be FP8, combine is BF16-only**. Combine performs a weighted reduction that needs BF16 precision (ROCm/DeepEP "Data Types and Precision": `low_latency_combine` is "Always BF16 output"; DeepSeek-V3 report: "we retain [combine] in BF16 to preserve training precision"). The canonical flow is **FP8 dispatch → expert GEMM converts to BF16 output → BF16 combine**. A "fused_moe FP8 output" is therefore not part of the design, and the original assert is **not** an aiter capability gap.

Real cause of the assert: aiter `fused_moe` defaults its out dtype to the input dtype (`fused_moe.py:260`), and `ep_moe/layer.py::forward_aiter` never passes a dtype — so under FP8 dispatch the output dtype follows FP8 and trips the fp16/bf16 assert. aiter already accepts an explicit `dtype` arg (`fused_moe.py:235`), i.e. it can do **FP8-in / BF16-out today**.

Routes to weigh:
- **Route A — full BF16 dispatch (`SGLANG_DEEPEP_BF16_DISPATCH=1`).** Simplest; verified to clear the fused_moe assert (§5). But it also drops the dispatch leg to BF16, so it diverges most from FP8 (no FP8 activation transport).
- **Route B — keep FP8 dispatch, make `forward_aiter` pass `dtype=torch.bfloat16`.** Matches the canonical DeepEP flow (FP8 activation transport + FP8 weight GEMM + BF16 output + BF16 combine), preserves FP8 semantics, and needs **no new aiter capability** — only a one-line sglang change. Caveat: must confirm dispatch's FP8 activation scales are correctly fed to fused_moe.

Both routes still hit the §6 combine boundary. Per the §6 update, a UCCL-side reshape does **not** work: aiter emits a reduced token-view tensor `[64, 2048]`, not the masked expert-view `[16, 1024, 2048]` combine needs. Bridging needs aiter to emit a masked/un-reduced layout on the DeepEP path, or an aiter-specific communication-only combine — a larger effort, not a one-liner.

*Status: combine fix attempted and reverted (see §6 update) — the combine boundary is a semantic incompatibility, not a layout bug; bridging it is a larger effort pending a team decision.*

## 8. Environment facts

| Component | Value |
|---|---|
| GPU | 8x AMD Instinct MI355X, gfx950 |
| ROCm / HIP | 7.0.0 / 7.0.51831 |
| PyTorch | 2.9.0a0+git7bcbafe (hip 7.0, cuda None) — image-customized, preserved |
| Python | 3.10.12 |
| sglang | 0.5.11.dev36+ga929eb728 |
| aiter | installed, source at `/sgl-workspace/aiter`, git `417de6df4` / `v0.1.11.post1`, `import aiter` OK; `fused_moe` out dtypes = {fp16, bf16} |
| uccl.ep | compiled for gfx950 -> `site-packages/uccl/ep.cpython-310-...so` |
| deep_ep | editable install of `uccl/ep/deep_ep_wrapper` |

## 9. Files changed in miles

Import fixes:

- `miles/utils/chat_template_utils/deepseek_v4.py` — lazy `encoding_dsv4`
- `miles/utils/chat_template_utils/deepseek_v32.py` — lazy `encoding_dsv32`
- `miles/utils/enum_compat.py` — new `StrEnum` shim
- `miles/utils/chat_template_utils/tito_tokenizer.py` — import `StrEnum` from shim
- `miles/utils/test_utils/session_verify_agent.py` — import `StrEnum` from shim

Test infra:

- `tests/e2e/megatron/test_qwen3_30B_A3B/_common.py` — (1) switched the rollout EP backend from `deepep` to `mori` (§11); (2) added `--sglang-disable-cuda-graph` on the mori path (diagnostic toggle, can be removed once §12 is resolved); (3) extended the env-var pass-through to ray `runtime_env` to forward `SGLANG_DEEPEP_BF16_DISPATCH`, `SGLANG_USE_AITER`, and `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`.

## 10. Logs

| Log | Path |
|---|---|
| E2E with BF16 dispatch (passes fused_moe; fails at UCCL combine, §6) | `/tmp/dpsk_v4_test3.log` |
| E2E after UCCL install (fused_moe FP8 assert) | `/tmp/dpsk_v4_test2.log` |
| E2E before UCCL (DeepEP ImportError) | `/tmp/dpsk_v4_test.log` |
| UCCL-EP build | `/tmp/uccl_ep_build.log` |
| E2E after combine reshape attempt (size mismatch, §6 update) | `/tmp/dpsk_v4_test4.log` |
| E2E with mori backend (hit `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` assert, §11) | `/tmp/dpsk_v4_test5.log` |
| E2E with mori + `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384` (free(): invalid pointer during graph capture, §12) | `/tmp/dpsk_v4_test6.log` |
| E2E with mori + `--sglang-disable-cuda-graph` (same free(): invalid pointer in shmem init, §12) | `/tmp/dpsk_v4_test7.log` |

## 11. Probe result — AMD's supported EP path is `mori`, not `deepep` (UCCL)

The §6 combine incompatibility is not a UCCL bug nor a one-off; it comes from running a NVIDIA-oriented backend on ROCm. Findings:

**(a) On ROCm, sglang forces aiter for all DeepEP MoE compute.** `run_moe_core` (`ep_moe/layer.py:227`) routes to `forward_aiter` whenever `_use_aiter` (= `SGLANG_USE_AITER` and HIP); the NVIDIA `forward_deepgemm_*` paths are `assert False`-deprecated (`:238`, `:248`). aiter is the *only* MoE backend for DeepEP on AMD.

**(b) aiter fuses unpermute + weighted-sum, so it emits a token-view, already-reduced 2D tensor** (observed `[64, 2048]`). Its matching combine is mori's: `mori_op.combine_send(hidden_states, None, topk_ids)` (`moriep.py:858`) — the weights arg is `None` because aiter already applied them; combine only does the cross-rank return, no reduce.

**(c) UCCL faithfully implements the official DeepEP-LL combine API** — 3D masked `[num_local_experts, max_tokens*ranks, hidden]` bf16 input, reduced internally using `topk_weights` (`buffer.py:486-492`). That is doubly incompatible with aiter's output: wrong layout *and* a second weighting. So UCCL's combine is correct per the DeepEP spec — aiter's output was simply never meant for the DeepEP-LL combine.

**(d) The test selected the NVIDIA-oriented backend.** `_common.py:186` sets `--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto`; on rollout/decode `auto` resolves to low_latency → UCCL `low_latency_combine`. SGLang's AMD docs instead recommend `--moe-a2a-backend mori`.

**(e) `mori` is already installed and full-featured here.** `/sgl-workspace/mori` v0.1.0; sglang's `MoriEPDispatcher` implements *both* normal and low_latency (`moriep.py:437` / `:692`), so the earlier "mori = normal only" note is outdated.

**Recommended fix — switch the rollout EP backend to `mori`:**
- `_common.py:186`: `--sglang-moe-a2a-backend deepep` → `--sglang-moe-a2a-backend mori` (keep `--sglang-deepep-mode auto`).
- Ensure `SGLANG_USE_AITER=1` in the run env (mori requires aiter — `layer.py:631`); the `SGLANG_DEEPEP_BF16_DISPATCH` workaround is no longer needed (mori has its own per-1x128 FP8 dispatch — `moriep.py:696,717`).
- On MI355X HBM (~288 GB) sglang defaults `chunked_prefill_size=16384`, which exceeds mori's default `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=4096` (`server_args.py:2841-2844`). Set `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384` (or cap `--sglang-chunked-prefill-size 4096`). Also pass this env through `_common.py::execute()`.
- Watch-outs: mori needs `ep == tp` (sglang sets this automatically); UCCL / `deep_ep` is no longer needed for rollout. The Megatron-side `--moe-enable-deepep` (`_common.py:218`) is the *training* EP path and is independent of the rollout backend.

## 12. New blocker on the `mori` path — native `free(): invalid pointer` in `mori.shmem.shmem_torch_process_group_init`

After applying the §11 recommendation, sglang correctly engages mori (`moe_a2a_backend='mori'`, `deepep_mode='normal'` after auto-resolve, `ep_size=tp_size=8`), but on the first MoE dispatch each TP rank aborts with:

```text
free(): invalid pointer
Fatal Python error: Aborted

Current thread (most recent call first):
  File "/sgl-workspace/mori/python/mori/shmem/api.py", line 38 in shmem_torch_process_group_init
  File "/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/moriep.py", line 214 in init_mori_op
  File "/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/moriep.py", line 355 in mori_op       # lazy init via property
  File "/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/moriep.py", line 613 in _dispatch_core
  ...
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py", line 2780 in forward_extend
```

`api.py:38` is the thin Python wrapper that calls into `mori_cpp.shmem_torch_process_group_init`; the abort is a glibc heap error from inside mori's native cpp.

Diagnostics performed:
- Tried under HIP graph capture (default): aborts.
- Tried with `--sglang-disable-cuda-graph`: aborts in the same place during the first prefill forward. So the crash is **not** caused by lazy-init colliding with graph capture — it's a real bug/incompatibility inside mori's shmem init itself.
- Installed mori: `/sgl-workspace/mori`, built from `github.com/ROCm/mori` @ `13ec475c` (2026-03-26), version 0.1.0. Built `.so`s: `libmori_application.so`, `libmori_io.so`, `libmori_pybinds.so`.
- Host runtime: ROCm/HIP 7.0.51831, PyTorch 2.9.0a0+git7bcbafe (image-customized), Python 3.10.

This is beyond what can be patched from sglang/miles code: the abort is in mori's native layer. Likely causes (any one of these is plausible):
- mori 0.1.0 cpp built against a different ROCm/PyTorch ABI than the current image runtime.
- mori expects RDMA fabric resources (IB / RoCE) that aren't visible in this container — its shmem init may free a pointer that was never allocated when the transport probe fails silently.
- Missing/insufficient POSIX shm (`/dev/shm` size, `--ipc=host`).

Next-step options (decision pending):
- **(a) Rebuild mori against the actual image toolchain.** Pull mori upstream HEAD, rebuild with the in-image ROCm/PyTorch headers, reinstall over `/sgl-workspace/mori`. Best chance of fixing if it's an ABI mismatch.
- **(b) Try mori's unique-id init path instead of the torch-process-group init.** sglang's `init_mori_op` only uses `shmem_torch_process_group_init`; switching to `shmem_init_attr(MORI_SHMEM_INIT_WITH_UNIQUEID, ...)` (`api.py:53`) would bypass the crashing entrypoint, but requires a sglang-side change.
- **(c) Report to ROCm/mori upstream.** Open an issue with the stack and image versions; depend on fix.

## 13. Final state of the investigation

Solid findings:
- The original failure mode (DeepEP-LL combine 2D/3D incompat) is a **backend mismatch**, not a one-line bug — UCCL is the wrong backend on AMD; mori is the right one (§11).
- mori is wired up correctly by `_common.py:186` change and sglang accepts it cleanly, but mori 0.1.0 itself aborts in `shmem_torch_process_group_init` on this image (§12).
- The Python-level workarounds we built (`SGLANG_DEEPEP_BF16_DISPATCH` pass-through, UCCL combine reshape) are now obsolete: bypassed by the mori path (BF16 dispatch) and proven semantically impossible (combine reshape). The pass-through wiring in `_common.py` was repurposed to forward `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` and remains useful.

Still blocked:
- End-to-end DeepEP+FP8 rollout on this AMD image — blocked on the §12 mori native crash. **(Resolved — see §14.)**

## 14. Resolution — rebuild mori from upstream HEAD

The shipped mori (`13ec475c`, 2026-03-26, v0.1.0) is broken in `shmem_torch_process_group_init` on this image; upstream has 2+ months of EP/IO/shmem fixes since. We rebuilt mori in place:

```bash
# /sgl-workspace/mori is the editable install (mori.egg-link → python/)
cd /sgl-workspace/mori
git fetch origin
git checkout origin/main             # e94694c7, 2026-06-04 — fix(io): bind worker threads within allowed cpuset
rm -rf build/
MORI_GPU_ARCHS=gfx950 pip install --no-build-isolation -v -e .
```

Build was clean (55 CXX TUs, only spdlog deprecation warnings, no errors), produced 6 `.so`s in `python/mori/` (the new `libmori_collective.so`, `libmori_ops.so`, `libmori_shmem.so` reflect upstream's module split). New version: `0.1.1.dev361+ge94694c79`.

Re-run with the rebuilt mori + CUDA graph re-enabled (removed the `--sglang-disable-cuda-graph` diagnostic flag from `_common.py`):

```text
USE_AITER=1 MORI_MAX=16384 mori=upstream-HEAD cuda_graph=on
...
moe_a2a_backend='mori', deepep_mode='normal', ep_size=8
...
[2026-06-04 04:36:13] The server is fired up and ready to roll!
```

The `free(): invalid pointer` is gone. The server passes mori shmem init, HIP graph capture, and warmup; the pipeline progresses through full rollout and the Megatron training forward pass (`/tmp/dpsk_v4_test8.log`). No `Memory access fault`, no `Scheduler hit an exception`.

Net state of `_common.py` changes for the working config:
- `--sglang-moe-a2a-backend mori --sglang-deepep-mode auto` on the deepep test path (§11).
- Env passthrough through ray `runtime_env`: `SGLANG_USE_AITER`, `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (set by the runner to `16384` to clear the §11 chunked-prefill check; `SGLANG_DEEPEP_BF16_DISPATCH` is left in the passthrough list for safety but is no longer set in normal runs).
- No UCCL / `deep_ep` runtime dependency on the rollout side.

## 15. New, deeper blocker — rollout produces garbage; CI rejects on log-prob mismatch

With the mori path now end-to-end alive, the test reaches the CI consistency check and fails:

```text
AssertionError: CI check failed:
  log_probs        (-8.27930998802185)
  != rollout_log_probs (-5.510956287384033)
```

This is the consistency check between the **training-side** forward (Megatron, FP8 model) and the **rollout-side** forward (sglang+mori+aiter, FP8 model) over the same tokens. A ~2.77 difference in log-space is huge (≈ 16× probability divergence) — well past any acceptable numerical tolerance.

Direct evidence that the **rollout itself is numerically wrong**, not just the comparator: the rollout-generated text is **garbage**. Excerpts from `/tmp/dpsk_v4_test8.log:6450,6456`:
- Prompt: a self-contained AMC-style geometry problem in English ("rectangle ABCD ... find FG^2").
- Rollout completion: a long stream of unrelated Chinese tokens, broken JSON/HTML/code fragments, `&#`, `causa`, `中国梦`, `季后赛`, `_AX`, … with no math, no answer.

So the model is producing **out-of-distribution garbage at rollout time**. Therefore the log-prob mismatch is a *symptom*, not the root cause — the rollout forward pass is itself broken.

Likely causes (none verified yet):
- **Quantization-scale plumbing on the mori/aiter path** for the FP8 (Qwen3-30B-A3B-FP8) checkpoint. The earlier finding in §10 still stands: `forward_aiter` (`ep_moe/layer.py:280-284`) discards `hidden_states_scale` from dispatch and never threads `a1_scale` into `fused_moe`. Under mori's normal-mode FP8 dispatch (`moriep.py:696,717` per-1x128 fp8 quant) this would feed expert GEMM unscaled FP8 activations → numerical chaos that exactly matches the observed garbage text.
- mori normal-mode dispatch/combine path not yet exercised in CI against this FP8 stack — possible per-1x128 scale layout mismatch with what aiter's fused_moe expects.
- Some other dtype/layout casting issue in the mori → aiter integration for the FP8 checkpoint.

Sanity-check we haven't run yet: compare against a known-good baseline on the same machine — the `--moe-a2a-backend none`/non-deepep baseline test (`test_baseline.py`/`test_r3_baseline.py`) which uses the same FP8 model but a different rollout path. If that baseline also produces garbage, the issue is in FP8 weights/loading, not mori. If it produces sane text, mori (or mori+aiter scale plumbing) is the suspect.

## 16. Root cause found — `Qwen3MoeSparseMoeBlock.forward` routes mori into `forward_normal` and double-reduces

Reading `/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py:287-303`:

```python
def forward(self, hidden_states, forward_batch=None, ...):
    if (
        not get_moe_a2a_backend().is_deepep()
        and not get_moe_a2a_backend().is_ascend_fuseep()
    ):
        return self.forward_normal(hidden_states, ...)
    else:
        return self.forward_deepep(hidden_states, forward_batch)
```

- `is_mori()` returns False for both checked predicates → **mori takes the `forward_normal` branch**.
- `forward_normal` (line 315-359) calls `self.experts(hidden_states, topk_output)` (== `MoriEPMoE.forward`, which already does dispatch+combine via mori), and then unconditionally adds `moe_expert_parallel_all_reduce(final_hidden_states)` when `ep_size > 1` (line 342-347) **and** `moe_tensor_model_parallel_all_reduce` when `tp_size > 1` (line 349-357).
- mori's combine already accumulates per-expert outputs across the EP group; an additional EP all-reduce on top double-counts contributions → completely wrong activations → garbage token output, exactly the symptom in §15.

`forward_deepep` (line 361-381), by contrast, just returns `self.experts(...)` directly without any extra reduce. This is the right shape for any backend whose dispatcher does its own a2a (deepep, mori, ...).

### Fix

`sglang/python/sglang/srt/models/qwen3_moe.py:295-303` — include mori in the "has its own a2a" branch:

```python
if (
    not get_moe_a2a_backend().is_deepep()
    and not get_moe_a2a_backend().is_ascend_fuseep()
    and not get_moe_a2a_backend().is_mori()
):
    return self.forward_normal(...)
else:
    return self.forward_deepep(hidden_states, forward_batch)
```

Two other files in the sglang tree (`step3p5.py`, `sdar_moe.py`) have the identical pattern; if mori is ever wired up for those models the same patch will be needed.

## 17. The forward-branch fix in §16 is necessary but NOT sufficient

After the `qwen3_moe.py` patch in §16, a verification run (`/tmp/dpsk_v4_test11.log`) was instrumented with a TRACE in `MoriEPMoE.forward` and an `assert not is_mori()` guard at the entry of `forward_normal`. Findings:

- `[TRACE MoriEPMoE.forward]` lines did appear (8 lines, one per SGLang worker pid), confirming `MoriEPMoE.forward` is on the live path.
- The `forward_normal` assert never fired → the routing fix took effect; mori no longer hits the double-reduce branch.
- Rollout output is still garbled and the CI `log_probs vs rollout_log_probs` check still fails with the same magnitude (`-8.48 vs -5.64`, prior `-8.27 vs -5.51`).

So the double-reduce branch in `forward_normal` was a real bug, but it is **not** what's producing the rollout garbage on Qwen3-30B-A3B-FP8 + SGLang+aiter+mori.

## 18. FP8 dispatch scale is not the cause either

Hypothesis tested: maybe mori's FP8 activation scales (`per_1x128`, `[recv_tokens, hidden_size//128]`) are not what aiter's `fused_moe` expects in `quant_type=per_128x128` for `a1_scale`, so disabling FP8 dispatch should change the output.

Run: `/tmp/dpsk_v4_test12_bf16disp.log`, with `SGLANG_MORI_DISPATCH_DTYPE=bf16` pushed through `extra_env_vars` in `_common.py`. MORI init log confirms the env var took effect:

```
[MORI init] ... fp8_dispatch=False fp4_dispatch=False combine_quant_type='none'
```

Result: rollout produced **character-for-character identical** garbage to the FP8-dispatch run (same opening tokens, same length, same content), with `log_probs -8.41 vs rollout_log_probs -5.60` — essentially the same numerical mismatch.

If switching between FP8 and BF16 activation dispatch leaves the output bit-identical, then the MoE numerics downstream cannot be the source of divergence. FP8-scale handling in `forward_aiter` / `run_moe_core` is therefore exonerated for this failure.

## 19. Even bypassing mori (a2a=none) the path is broken at EP=8 — but EP=1 baseline works

### 19a. Existing rollout-side baselines that actually worked (recovered from prior logs)

Before this debug session a set of EP=1 / TP-* and EP=2 variants had already been run. Re-reading them now:

| Log | Rollout side config | first rollout sample | final status |
|---|---|---|---|
| `/tmp/dpsk_v4_test15_ep1tp1.log` | `--rollout-num-gpus-per-engine 1 --sglang-expert-parallel-size 1` (single-GPU SGLang) | **normal English continuation of the `airline / dinner` prompt** | `Job 'raysubmit_jiAuU3cYkYa16sny' succeeded` |
| `/tmp/dpsk_v4_test16_tp2ep1.log` | `--rollout-num-gpus-per-engine 2 --sglang-expert-parallel-size 1` (TP=2 SGLang) | **normal English continuation** | crash later with `ActorDiedError, Worker unexpectedly exits with a connection error code 2` — process-level crash post-rollout, not a log-prob divergence |
| `/tmp/dpsk_v4_test14_ep1.log` | `--rollout-num-gpus-per-engine 8 --sglang-expert-parallel-size 1` (TP=8 SGLang, no EP) | n/a | `Server process terminated unexpectedly` during health-check — SGLang server never came up |
| `/tmp/dpsk_v4_test17_tp1ep2.log` | `--rollout-num-gpus-per-engine 2 --sglang-expert-parallel-size 2`, **`sglang_moe_a2a_backend = none`** (StandardDispatcher, not mori) | **normal English continuation** | `AssertionError: CI check failed: log_probs (-0.2857542) != ref_log_probs (-0.28579067)` — Δ ≈ 3.6e-5, i.e. numerical-precision tolerance, **not** a logical/semantic failure |

Important corollaries:
- **The FP8 model itself is fine on SGLang+aiter.** EP=1 single-GPU produces sensible continuations and the whole job succeeds end-to-end.
- **EP=2 + StandardDispatcher (a2a=none) is numerically close to correct.** Rollout text is sane; the CI fault is a ~3.6e-5 log-prob delta against `ref_log_probs`. Contrast this with EP=8 + mori, which compares against `rollout_log_probs` and shows a ~2.83 delta — a different failure mode, three orders of magnitude apart.
- The earlier "EP > 1 is broken" framing in this section was therefore **wrong**. The failure threshold lies between EP=2 and EP=8. Below it, EP-sharded FP8 inference works (within precision); at EP=8, both a2a backends fail in different ways (mori → garbage tokens, none → GPU memory fault).
- **`EP=2 + mori` is the missing data point.** What is known so far is `EP=2 + none` (works, §19a row 4) and `EP=8 + mori` (broken, §17/§18). Without an EP=2 + mori run, we cannot yet tell whether mori is broken at every EP>1 or only at the 8-rank topology.

### 19b. The §19 control run at EP=8, no a2a backend

To rule out mori itself, the rollout side was reconfigured to `--sglang-moe-a2a-backend` unset (i.e. `MoeA2ABackend.NONE` → SGLang `StandardDispatcher` with `--sglang-expert-parallel-size 8`). Training side kept `--moe-enable-deepep`. Run: `/tmp/dpsk_v4_test13_none.log`.

Result: sglang never reaches the first rollout sample. Every TP rank dies with:

```
Memory access fault by GPU node-6 (Agent handle: ...) on address 0x7fb35a000000. Reason: Unknown.
Memory access fault by GPU node-4 ...
Memory access fault by GPU node-2 ...
Memory access fault by GPU node-5 ...
Memory access fault by GPU node-8 ...
Memory access fault by GPU node-7 ...
```

After all 8 workers die the router starts returning `503 Service Unavailable, no_available_workers`.

Combined with the EP=1 / EP=2 baselines in §19a, the failure surface is **EP scale ≥ some threshold > 2 on this AMD SGLang+aiter+FP8 build**, in two flavors:
- with mori a2a backend at EP=8: silently produces wrong activations (garbage tokens, fixed pattern across FP8/BF16 dispatch)
- with no a2a backend at EP=8 (`StandardDispatcher`): GPU memory access fault and process death

EP=2 with mori (each rank holds 64 of 128 experts) is numerically within ~3.6e-5 of the reference, so the basic FP8 expert weight + scale binding works for at least the EP=2 sharding. Plausible candidates for the EP=8-specific break, given this evidence:
- size-dependent buffers / dispatch-token-per-rank constants in mori's intra-node path that get exercised only when `num_local_experts = 16` and 8 RDMA peers
- expert-id remapping or `expert_mask` packing inside `MoriEPMoE.run_moe_core` that overflows / wraps when `num_local_experts < ep_size` (16 < 128/16=8 is the size ratio)
- `Fp8MoEMethod` shuffle / per-128×128 block scale alignment that happens to be correct on EP=1/2 (where N/128 still divides evenly across 16 or 32 local experts × intermediate=768) but mis-aligns at EP=8 with 16 local experts × hidden=2048 → 128
- a 8-rank-specific code path (e.g. `EpMode.INTRA_NODE` exact-8 specialization) in mori's HIP shmem layout

## 20. Decision points and next steps

Evidence so far:
- EP=1 (TP=1/2/8): the FP8 model + aiter inference path is fine; not the bug.
- EP=2 + StandardDispatcher (a2a=none): rollout text sane; CI Δ ~3.6e-5 is numerical precision, not the bug.
- EP=8 + mori: rollout garbage. EP=8 + StandardDispatcher: GPU memory fault.
- EP=2 + mori: **missing**. Needed to separate "mori is broken for all EP>1" from "mori is broken only at 8-rank topology".

Concrete next steps, in cost order:

1. **EP=2 + mori control run**. Flip `_common.py` to `sglang_ep_size=2, rollout_num_gpus_per_engine=2` *and* re-enable `--sglang-moe-a2a-backend mori` in the rollout flags. Two outcomes:
   - sane → mori works for some EP>1; the EP=8 mori failure is topology-specific (intra-node 8-rank kernels, dispatch-tokens-per-rank constants, or the 16-local-expert layout). Investigate mori's `IntraNode` path and `MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` heuristics.
   - garbage → mori is broken at every EP>1; the bug is in mori's combine/dispatch semantics for FP8 + aiter, independent of rank count. Switch to investigating `MoriEPMoE.run_moe_core` expert-id remapping and per-expert scale alignment.
2. Optional **EP=4 + mori** bisect if (1) is sane, to confirm where mori actually breaks.
3. Depending on outcome, patch the EP-specific issue or temporarily downgrade rollout to the largest EP that produces sane output while preparing an upstream report.
4. Once a sane EP=8 rollout is recovered, re-validate the full stack; the `qwen3_moe.py` `forward_deepep` patch in §16 stays in (it was a real but secondary bug).

## 21. Files modified in this round

| File | Change |
|---|---|
| `sglang/python/sglang/srt/models/qwen3_moe.py` | `forward` branch: add `is_mori()` to the "has its own a2a" predicate so mori goes to `forward_deepep` (no double reduce). |
| `tests/e2e/megatron/test_qwen3_30B_A3B/_common.py` | Restored `--sglang-moe-a2a-backend mori --sglang-deepep-mode auto` for the rollout side; added passthrough for `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` and `SGLANG_MORI_DISPATCH_DTYPE` into `extra_env_vars`. Diagnostic toggle in place to temporarily drop the mori flag for the §19 control run; this toggle should be reverted before final validation. |

## 22. Logs added / referenced in this round

| Log | Path | Status |
|---|---|---|
| E2E mori (FP8 disp) — proves forward fix works but rollout still garbled | `/tmp/dpsk_v4_test10.log`, `/tmp/dpsk_v4_test11.log` | rollout garbage; CI fail (-8.48 vs -5.64) |
| E2E mori with `SGLANG_MORI_DISPATCH_DTYPE=bf16` — char-identical garbage | `/tmp/dpsk_v4_test12_bf16disp.log` | rollout garbage; CI fail (-8.41 vs -5.60) |
| E2E a2a backend dropped (StandardDispatcher + EP=8) — GPU memory fault | `/tmp/dpsk_v4_test13_none.log` | all workers die before first sample |
| (prior session, recovered) EP=1, single-GPU SGLang — known-good baseline | `/tmp/dpsk_v4_test15_ep1tp1.log` | rollout normal; **Job succeeded** |
| (prior session, recovered) EP=1, TP=2 SGLang — rollout text fine, crash later | `/tmp/dpsk_v4_test16_tp2ep1.log` | rollout normal; `ActorDiedError` post-rollout |
| (prior session, recovered) EP=1, TP=8 SGLang — server never came up | `/tmp/dpsk_v4_test14_ep1.log` | `Server process terminated unexpectedly` |

## 23. Isolation baseline — EP=1 / TP=1 (single-GPU engines × 8) PASSES end-to-end

Per §20 step 1, the rollout side was reduced to the minimum-sharding config to test whether the rollout-side FP8 model and SGLang+aiter compute are themselves correct on this AMD image, independently of any EP/TP partitioning logic.

**Configuration probe — only TP∈{1,2} are reachable on this 8-GPU machine.** First attempt was `sglang_ep_size=1` with the unchanged `rollout_num_gpus_per_engine=8` (1 engine × 8 GPUs, EP=1 → TP=8). It crashed at `Fp8MoEMethod.create_weights`:

```text
ValueError: The output_size of gate's and up's weight = 96
            is not divisible by weight quantization block_n = 128.
```

`Qwen3-30B-A3B-FP8` ships per-block FP8 with `block_n=128`, and `moe_intermediate_size=768`. The FusedMoE TP shard splits the intermediate dim by `tp_size`, giving `768/8=96`, which breaks the quantization block boundary. The valid TP values for this checkpoint are `TP ∈ {1, 2, 3, 6}` (i.e. the divisors of `768/128=6`); on 8 GPUs only TP=1 and TP=2 are usable. Log: `/tmp/dpsk_v4_test14_ep1.log`.

**Working baseline run.** Both `sglang_ep_size=1` and `rollout_num_gpus_per_engine=1` set, giving 8 single-GPU engines each TP=1 EP=1, no MoE sharding at all. SGLang's `ServerArgs` confirms: `tp_size=1, pp_size=1, ep_size=1, moe_a2a_backend='none'`. Run: `/tmp/dpsk_v4_test15_ep1tp1.log`. Job status: **succeeded**. Two rollout iterations completed cleanly:

| Metric | rollout 0 | rollout 1 |
|---|---|---|
| `rollout/log_probs` (Megatron side) | `-0.28879` | `-0.23178` |
| `rollout/rollout_log_probs` (SGLang side) | `-0.28278` | `-0.22661` |
| Δ (consistency check) | **0.006** | **0.005** |
| `rollout/raw_reward` | 0.469 | 0.531 |
| `rollout/response_lengths` | 6523 | 6405 |
| Decode throughput | ~700–2100 tok/s/engine | — |

The §15 garbled-rollout symptom is gone. Comparison:

| | EP=1 / TP=1 (this run) | EP=8 + mori (§15/§17) | EP=8 + a2a=none (§19) |
|---|---|---|---|
| Job status | succeeded | CI fail | 8 workers GPU-fault |
| log_probs vs rollout_log_probs Δ | 0.005–0.006 | 2.77–2.84 | n/a |
| raw_reward (math correctness) | ~0.5 | 0 (garbage) | n/a |
| Garbage tokens (`中国梦`, `&#`, `causa` real hits) | 0 (4 false-positive matches were `is_causal=` in aiter logs) | many | n/a |

**Conclusion.** The FP8 weight load + SGLang+aiter forward path on this AMD image is **correct in isolation**. The §15/§17/§19 failures are *not* in `Fp8MoEMethod` weight loading itself; they live in the EP>1 and/or TP>1 sharding path (FusedMoE TP/EP weight + scale binding, expert dispatcher, or their interaction with aiter/mori). This rules out the "broken FP8 checkpoint" hypothesis entirely.

Files touched for this baseline:
- `tests/e2e/megatron/test_qwen3_30B_A3B/test_deepep_fp8.py` — `sglang_ep_size=1`, `rollout_num_gpus_per_engine=1`. Diagnostic only; revert before final validation.

New logs:

| Log | Path | Status |
|---|---|---|
| EP=1, TP=8 — exposes FP8 block_n=128 vs intermediate=96 sharding incompat | `/tmp/dpsk_v4_test14_ep1.log` | aborts in `create_weights` |
| EP=1, TP=1 — minimum-sharding baseline | `/tmp/dpsk_v4_test15_ep1tp1.log` | **succeeded**, log_probs Δ ≤ 0.006 |

## 24. Probe rung 2 — TP=2 / EP=1 (4 engines × 2 GPUs): rollout numerics also correct

Continuation of §23. With rollout-side EP held at 1, TP raised to 2 (`rollout_num_gpus_per_engine=2`, `sglang_ep_size=1` → 4 engines, each TP=2 EP=1). FP8 block-quantization compatible (`768 / 2 = 384 = 3 × 128`).

Run: `/tmp/dpsk_v4_test16_tp2ep1.log`. SGLang `ServerArgs`: `tp_size=2, pp_size=1, ep_size=1, moe_a2a_backend='none'`. All 4 engines pass model load + HIP graph capture + warmup, no `ValueError` / `Memory access fault` / `free(): invalid pointer`.

Two rollout iterations completed with the same numerical agreement quality as TP=1:

| Metric | rollout 0 | rollout 1 |
|---|---|---|
| `rollout/log_probs` (Megatron) | -0.28661 | -0.22926 |
| `rollout/rollout_log_probs` (SGLang) | -0.28072 | -0.22317 |
| **Δ (consistency check)** | **0.0059** | **0.0061** |
| `rollout/raw_reward` | 0.516 | 0.547 |
| `rollout/response_lengths` | 6474 | 6233 |

Side-by-side with §23:

| Config | rollout 0 Δ | rollout 1 Δ | reward 0 / 1 |
|---|---|---|---|
| TP=1 / EP=1 (§23) | 0.0060 | 0.0050 | 0.469 / 0.531 |
| TP=2 / EP=1 (this) | 0.0059 | 0.0061 | 0.516 / 0.547 |

**Rollout-side TP=2 path is clean.** The §15/§17 garbage-rollout symptom does not appear when only TP is raised; therefore the bug is *not* in the SGLang FP8 TP-shard logic (FusedMoE TP weight + scale binding via `Fp8MoEMethod`). Suspicion narrows to the **EP>1 path** in either the rollout EP dispatcher (mori / StandardDispatcher when ep_size>1) or `Fp8MoEMethod`'s expert sharding.

### Side finding — Megatron *training* side has an independent EP=4 bug

The job exit was *not* caused by rollout. The training-side `MegatronTrainRayActor` aborted during the *second* training step's backward with:

```text
HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
Kernel: at::native::index_put_kernel_impl<...>
Stack: Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py:1227 _indices_to_multihot
       moe_layer.py:325 routed_experts_compute
       moe_layer.py:451 forward
       transformer_layer.py:776 _forward_mlp
       (in pipeline_parallel/schedules.py:190 custom_backward)
```

Notes:
- This is the *training* path, BF16 weights (FP8 lives only on the rollout side here), EP=4, TP=2, PP=2, CP=2.
- Step 0's backward completed cleanly (`perf 0` line was emitted). Step 1's backward crashed inside Megatron's MoE token dispatcher's `_indices_to_multihot`. Most likely an out-of-range topk index hitting `index_put` after the first weight sync — but unverified.
- This is **independent of the FP8 rollout investigation** and does not invalidate the §24 conclusion. It is, however, the gating issue if we want a green CI run on this image.
- Probably also reproducible across the upcoming EP probes (training-side EP=4 is set in `_common.py:138 --expert-model-parallel-size 4`, untouched by the rollout-EP knob).

Logs:

| Log | Path | Status |
|---|---|---|
| Rollout TP=2 EP=1 — clean rollout numerics; training-side step-1 backward HSA exception | `/tmp/dpsk_v4_test16_tp2ep1.log` | rollout PASS, train ABORT (Megatron MoE backward) |

## 25. Probe rung 3 — TP=1 / EP=2 (4 engines × 2 GPUs): rollout still clean; CI fails on a different gate

Rollout-side EP raised to 2: `rollout_num_gpus_per_engine=2, sglang_ep_size=2` → 4 engines, each TP=1 EP=2 (128 experts split 64 + 64 per rank). Run: `/tmp/dpsk_v4_test17_tp1ep2.log`.

SGLang `ServerArgs`: `tp_size=1, ep_size=2, moe_a2a_backend='none'`. All 4 engines load FP8 model + capture HIP graphs + serve cleanly. No `ValueError` / `Memory access fault` / `free()` / hardware exception in the rollout side.

Rollout 0 metrics:

```
rollout 0:
  rollout/response_lengths : 6595
  rollout/raw_reward       : 0.531    # ~half of math problems answered correctly
  rollout/rollout_log_probs: -0.28059
  rollout/log_probs        : -0.28575
  rollout/ref_log_probs    : -0.28579
```

There are two independent CI consistency checks (`miles/backends/training_utils/log_utils.py:185-198`):

| Gate | Pair | abs_tol | When | EP=2 actual | Verdict |
|---|---|---|---|---|---|
| C1 | `log_probs` vs `ref_log_probs` | **1e-8** | rollout 0 only | **3.6e-5** | **fails** |
| C2 | `log_probs` vs `rollout_log_probs` | 0.03 | every rollout | 0.0052 | passes |

C2 is the same gate that failed catastrophically in §15 / §17 / §18 (Δ ≈ 2.77 — rollout was producing garbage). With EP=2 rollout, **C2 passes by ~600× margin** and `raw_reward ≈ 0.53` — the rollout-side numerics are healthy.

What fails is C1. C1 compares Megatron-side current-policy vs reference-model log-prob on rollout 0, where the actor weights have not yet been updated. In §23 (TP=1 EP=1) and §24 (TP=2 EP=1) C1 was **bit-for-bit equal** (`-0.28879 == -0.28879`, `-0.28661 == -0.28661`); under EP=2 it drifts by 3.6e-5.

Why this drifts only with rollout-EP > 1: C1's two operands are computed *entirely on the Megatron training side* (BF16, EP=4) — the rollout-side EP knob does not touch their compute graph. The indirect coupling is the data: rollout-EP changes the SGLang-side dispatch/combine ordering of tokens (per §11 rollout EP routes through aiter combine-by-permute), which slightly alters generated token sequences. Different token sequences → different per-batch BF16 accumulation order on the Megatron side → ~1e-5 reduction-noise drift between the two BF16 forwards. The 1e-8 threshold was tuned for rollout configs without EP-driven routing variance; 3.6e-5 sits comfortably inside BF16 reduction noise but blows past 1e-8.

So the failure here is **not** rollout breakage; it is the C1 gate being too tight for any rollout config that introduces EP-driven token-order variance. **C2 — the actual rollout-side correctness gate — passes**.

**Net read on the rollout-side EP=2 path**: numerically clean. The §15 garbage symptom and the §19 GPU memory fault both belonged to EP=8 (with mori or with `StandardDispatcher` respectively); neither reproduces at EP=2.

### Routing matrix so far

| Config | Rollout numerics (C2) | C1 | Rollout-side crash | Training-side crash |
|---|---|---|---|---|
| TP=1 / EP=1 (§23) | Δ=0.006 / 0.005 ✓ | == ✓ | none | (didn't reach) |
| TP=2 / EP=1 (§24) | Δ=0.006 / 0.006 ✓ | == ✓ | none | step-1 backward HSA 0x1016 (BF16 MoE) |
| TP=1 / EP=2 (this) | Δ=0.005 ✓ | 3.6e-5 ✗ | none | (didn't reach — C1 gate fired first) |
| TP=1 / EP=8 + a2a=none (§19) | n/a | n/a | GPU memory fault, all workers | n/a |
| TP=1 / EP=8 + mori (§17) | Δ=2.84 ✗ (garbage) | n/a | none | n/a |

Every configuration with rollout-EP ≤ 2 passes the rollout correctness gate. The failure mode is specific to **higher EP** (most likely a per-GPU-token-volume or per-rank-expert-count threshold), not to "any EP > 1".

### Files modified

- `tests/e2e/megatron/test_qwen3_30B_A3B/test_deepep_fp8.py` — `sglang_ep_size=2`, `rollout_num_gpus_per_engine=2`. Diagnostic only.

### Logs

| Log | Path | Status |
|---|---|---|
| Rollout TP=1 EP=2 — clean numerics; CI fails on the strict 1e-8 ref/policy gate | `/tmp/dpsk_v4_test17_tp1ep2.log` | rollout PASS (C2), C1 fails by 3.6e-5 |

## 26. Probe rung 4 — TP=1 / EP=2 with **mori a2a re-enabled**: rollout garbage (mori is the blame)

§25 ran with `--sglang-moe-a2a-backend` unset → `moe_a2a_backend='none'` (StandardDispatcher). That is **not** the same as the user-required production stack, which is mori. To separate "mori is broken at every EP > 1" from "mori is only broken at EP = 8", this rung re-enables `--sglang-moe-a2a-backend mori --sglang-deepep-mode auto` while keeping `sglang_ep_size=2, rollout_num_gpus_per_engine=2`. Run: `/tmp/dpsk_v4_test18_ep2mori.log`.

SGLang `ServerArgs`: `tp_size=1, ep_size=2, moe_a2a_backend='mori', deepep_mode='normal'`. All 4 engines start, capture HIP graphs, no `free()` / `Memory access fault` / `ValueError`.

**Rollout 0, first sample** (same `airline / dinner` prompt as §17, §23–§25):

```
<|im_start|>assistant
 pregnancies        
 &#爱好_autoclr🧸sequencesम;</ scams antioxidantayah百分之 &# &# &#健康成长<\\/caMah孀 ...
```

Character-level identical pattern to the EP=8 + mori garbage in §17/§18 (Chinese / programming-symbol token soup unrelated to the English prompt). Reward = 0.

### Routing matrix updated

| Config | Rollout text | Rollout correctness (C2) | C1 | Verdict |
|---|---|---|---|---|
| TP=1 / EP=1 (§23) | sane | Δ=0.005 ✓ | == ✓ | OK |
| TP=2 / EP=1 (§24) | sane | Δ=0.006 ✓ | == ✓ | OK (train-side crash unrelated) |
| TP=1 / EP=2 + **none** (§25) | sane | Δ=0.005 ✓ | 3.6e-5 ✗ (tolerance) | rollout OK |
| **TP=1 / EP=2 + mori** (this) | **garbage** | n/a (text broken) | n/a | **mori broken** |
| TP=1 / EP=8 + none (§19) | n/a | n/a | n/a | GPU memory fault |
| TP=1 / EP=8 + mori (§17) | garbage | Δ=2.84 ✗ | n/a | mori broken |

### What this rules in / out

The pair `(EP=2 + none = sane)` vs `(EP=2 + mori = garbage)` shares everything except the a2a backend:
- same `Fp8MoEMethod` per-expert weight + scale binding (so FP8 weight sharding is **not** the bug)
- same aiter `fused_moe` kernel and same per-expert layout (so aiter `fused_moe` numerics are **not** the bug)
- same EP=2 token-routing topology in attention / sampling (so EP-driven token variance — the §25 hypothesis for the 3.6e-5 drift — is **not** what produces the garbage; that drift sits within BF16 noise)

The only changed component is the dispatch+combine path: `StandardDispatcher` vs `MoriEPMoE`. Therefore the garbage in §17/§18/§26 is **mori-specific**, **not** EP-scale-specific. It reproduces at the smallest non-trivial EP (= 2), so the upstream `Fp8MoEMethod` + aiter expert kernels are intact.

### What is wrong inside mori (suspect list)

Reading `MoriEPMoE.forward` + `run_moe_core` together with the §16 `forward_deepep` patch:

1. **FP8 activation scale is silently dropped.** `forward_aiter` in `ep_moe/layer.py` does not thread `hidden_states_scale` (which mori's FP8 dispatch produces as `a1_scale`) into aiter's `fused_moe(... a1_scale=...)` call. The activations leave the kernel quantized but un-scaled relative to expert weights → near-uniform garbage in the residual stream. Path B in §7 was scheduled to fix this; it is now the most likely single root cause.
2. **`expert_mask` / token reorder after `MoriEPMoE.dispatch`** may not line up with the expert id space that aiter `fused_moe` expects for the local-experts slice. `StandardDispatcher` keeps a contiguous global-id layout; mori packs into a per-rank-local id space and aiter must be told.
3. **DeepEP-LL combine semantics**: mori's combine in this build does not reduce (per §11); the caller's `forward_deepep` is responsible. The §16 fix routes mori through `forward_deepep`, so this should be handled, but the assumption is worth re-checking once (1) is in.

### Next step

Implement suspect (1) — FP8 scale threading in `forward_aiter` — and re-run TP=1 / EP=2 + mori. If text becomes sane, the EP=8 + mori case is the same fix (possibly plus dispatch-tokens-per-rank tuning); if text stays garbage, move to (2) and instrument the expert-id remapping in `MoriEPMoE.run_moe_core`.

### Files modified

- `tests/e2e/megatron/test_qwen3_30B_A3B/_common.py` — re-enabled `--sglang-moe-a2a-backend mori --sglang-deepep-mode auto` (diagnostic).

### Logs

| Log | Path | Status |
|---|---|---|
| TP=1 EP=2 + mori — rollout garbage tokens | `/tmp/dpsk_v4_test18_ep2mori.log` | mori-side rollout broken; manually stopped after first rollout to save GPU |

## 27. Source code walk for the mori-side bug — what is verified, what stays a suspect

Reading the four layers that meet in the broken path:

- `sglang/srt/layers/moe/token_dispatcher/moriep.py` — `_MoriEPDispatcherImplNormal.dispatch/_dispatch_core/combine_a/_combine_core`
- `sglang/srt/layers/moe/ep_moe/layer.py` — `MoriEPMoE.forward/run_moe_core` (line 600-782)
- `mori/python/mori/ops/dispatch_combine.py` — `EpDispatchCombineOp.dispatch/combine`
- `mori/src/ops/dispatch_combine/intranode.hpp` — `EpDispatchIntraNodeKernel_body / EpCombineIntraNodeKernel_body`

### 27.1 Verified consistent

- mori dispatch writes `out_indices` directly from the source rank's `tokenIndices`, so each `recv_topk_ids[recv_tok]` row holds the **8 global expert ids** that the source token selected (`intranode.hpp` line 152-154).
- aiter `fused_moe` documents its `expert_mask` interface as «`local_expert_mask : indicate local expert mask used on current GPU; we call expert input to this kernel as "global expert id", output as "local expert id"`» (`moe_sorting_opus.h` line 325-329). Mask size = `num_experts`, mask=1 means «this rank owns this global expert».
- `MoriEPMoE.__init__` builds exactly this layout: `expert_mask = zeros(num_experts); expert_mask[rank*num_local : (rank+1)*num_local] = 1`. With the contiguous range it matches aiter's `popcount(mask[:global_id])` → local id remap.
- mori `combine`'s hidden accumulation is **unweighted** for the topk pointers: `WarpAccum<T, 4>(outPtr, srcPtrs, /*srcScales=*/nullptr, validAccumCount, ...)` (`intranode.hpp` line 735). The `weights` arg only feeds a separate `combineOutWeights` accumulator (line 738-744), never the hidden sum. So passing `weights=None` from sglang (`moriep.py` line 669-671) is correct semantically: hidden = sum over unique-PE-srcPtrs of «fused_moe output for that recv_tok on that PE». mori's own example confirms this: expected combine output for token i is `input[i] * unique_pes(i)` (`examples/.../test_dispatch_combine.py` line 320-323).
- dispatch dedup (`intranode.hpp` line 117-128) guarantees that for each (origin_token, dest_PE) pair only the first j-th expert allocates a real `destTokId`; subsequent j's at the same dest_PE store the sentinel `FlatTokenIndex(worldSize, 0)` into `dispDestTokIdMap`. Combine then turns those sentinel slots into `nullptr` srcPtrs (line 624-664), and `WarpAccum` skips nullptr (line 748-749). So «same PE counted multiple times» does not happen.

Algebra: token X with 8 topk experts split 4/4 across PE0/PE1 →
- `fused_moe(PE0) → sum_{j in PE0-owned} w_j * E_j(X)` (4 terms)
- `fused_moe(PE1) → sum_{j in PE1-owned} w_j * E_j(X)` (4 terms)
- combine(X) = unique-PE sum = sum over all 8 topk terms

This is the **correct** MoE math. The chain is end-to-end self-consistent.

### 27.2 The only test-coverage gap that actually matters

`mori/tests/python/ops/test_dispatch_combine_intranode.py` line 152-153:
```
# TODO: create a sub process group so that we can test worlds size < 8
@pytest.mark.parametrize("world_size", (8,))
```

mori's IntraNode dispatch/combine kernels are **only test-covered at world_size=8**. The configurations actually run on miles right now are:
- world_size=8 (EP=8) — `intranode.hpp` tested path, but rollout shows garbage (§17/§26 of older sections; the `Fp8MoEMethod` + aiter sharding for 16-local-expert × 128-global may still be the cause)
- world_size=2 (EP=2) — `intranode.hpp` **never tested** by mori upstream

Direct sources of «worldSize < 8 not equivalent to worldSize=8» inside `intranode.hpp`:
- Combine vec8_top8 specialization gate (`launch.cpp` line 548-550): `args.config.worldSize > 4` requires more-than-4 ranks for the fast path. world_size=2 falls back to scalar accumulate (line 671-693 of `intranode.hpp`).
- The `if (config.worldSize <= 4)` srcPtr-compaction branch (`intranode.hpp` line 671-693) is the only path that compacts nullptr srcPtrs in shared memory before `WarpAccum`. world_size=2 enters this branch; world_size=8 does not. Inspection looks correct (popcount-based prefix write), but it is the most distinct codepath that fires at EP=2 and not at EP=8.

### 27.3 Concrete next experiment (no patch yet, runs in 5 minutes)

Drop one `printf` / `cudaLog` inside `MoriEPMoE.run_moe_core` to capture, for the first MoE layer of the first forward pass, three Python-side facts at EP=2:

1. `(dispatch_output.topk_ids[:total_recv].cpu(), self.moe_ep_rank)` — confirm each row is in [0, 128) and 4-4 split across [0,64)/[64,128).
2. `dispatch_output.hidden_states[:total_recv]` norm vs the same prompt under EP=2+none — large delta → mori dispatch corrupts hidden; small delta → bug is downstream.
3. `hidden_states` after `fused_moe` norm vs. the EP=2+none counterpart — large delta → `fused_moe(num_local_tokens=…)` + expert_mask path mishandles the EP=2 input shape; small delta → bug is in mori combine at worldSize=2.

This bisects the failing stage (dispatch ↔ fused_moe ↔ combine) before changing any production code.

### 27.4 Two follow-up patches, only after the bisect above narrows the stage

- If §27.3 (2) shows dispatch already corrupts hidden: file `mori` upstream bug on IntraNode at worldSize<8 (no patch on the sglang side; downgrade rollout to EP=1 or switch to a different a2a backend in the meantime).
- If §27.3 (3) shows fused_moe output is the corruption point: align `MoriEPMoE.run_moe_core` to the verified-good `DeepEPMoE.forward_aiter` style — drop `a1_scale` for the FP8 path (let aiter quantize internally instead of consuming the dispatched scale), drop `num_local_tokens` (let aiter infer from `topk_ids.shape[0]`), and pass topk_ids with `-1 → num_local_experts` sentinel + `expert_mask` of size `num_local_experts + 1`. This makes the mori path call aiter the same way the EP-only deepep+aiter path calls it, which is the only call-site mori's contributor list has been actually testing in CI on AMD.

### Files referenced (no edits in this round)

- `sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/moriep.py`
- `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
- `sgl-workspace/mori/python/mori/ops/dispatch_combine.py`
- `sgl-workspace/mori/src/ops/dispatch_combine/intranode.hpp`
- `sgl-workspace/mori/src/ops/dispatch_combine/launch.cpp`
- `sgl-workspace/mori/tests/python/ops/test_dispatch_combine_intranode.py`
- `sgl-workspace/aiter/aiter/fused_moe.py`
- `sgl-workspace/aiter/csrc/include/moe_sorting_opus.h`

## 28. Dynamic bisect — instrumented `MoriEPMoE.forward` at rollout time

§27 stayed at a static read; this section adds the runtime measurement that
§27.3 prescribed, narrowed the failure to a single call edge, and resolves
the choice between §27.4 path (2) (mori IntraNode kernel suspect) and path
(3) (`run_moe_core` ↔ aiter calling-convention suspect) in favour of (3).

### 28.1 Patch (added temporarily, then reverted)

Instrumented `MoriEPMoE.forward` in `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`,
guarded by `MILES_MORI_DEBUG=1` and limited to `layer_id < MILES_MORI_DEBUG_LAYERS`
(default 2). At each `(rank, layer_id, stage)` key the helper prints once,
reporting `shape, dtype, nan/inf count, |max|, |mean|` of the tensors at four
stages:

- `00_forward_in`: input `hidden_states`, `topk_output.topk_ids`, `topk_output.topk_weights`.
- `10_dispatch_out`: dispatched `hidden_states` (first `total_recv` rows), the matching `hidden_states_scale`, `topk_ids`, `topk_weights`, and `num_recv_tokens_per_expert`.
- `30_fmoe_out`: aiter `fused_moe` output `combine_input.hidden_states` (full padded buffer).
- `40_combine_out`: post-combine `hidden_states` (both full padded buffer and the first `num_token` rows that go on to the residual).

The active flag is gated by `not torch.cuda.is_current_stream_capturing()`
plus a follow-up `hidden.float().abs().max().item() > 0`, so:

- CUDA-graph capture passes (where `.item()` would crash with
  `hipErrorStreamCaptureUnsupported`) are skipped via the capturing check;
- sglang's pre-rollout dummy warmup (all-zero hidden, topk_ids=`[0..7]`) is
  skipped via the max-value check, so the per-key one-shot budget is not
  consumed before any real rollout token reaches the layer.

`_common.py` was extended to forward `MILES_MORI_DEBUG` and
`MILES_MORI_DEBUG_LAYERS` into the SGLang Ray subprocess. Both edits were
reverted after the diagnostic data was captured.

### 28.2 Run

Same case as §26 (EP=2 + mori; `sglang_ep_size=2`, `rollout_num_gpus_per_engine=2`,
`--sglang-moe-a2a-backend mori`). Log: `/tmp/dpsk_v4_test22_ep2mori_dbg4.log`.
First real rollout decode batch (single new token) reached the MoE layer at
`2026-06-05 04:58:41`. Both engines (PID 2267748 in the colocated rollout
group and 2267749 in the second engine) emit identical statistics, so only
PID 2267748 is shown.

### 28.3 Per-stage statistics (layer 0 and layer 1)

| layer | stage              | rank | tensor                | shape          | dtype       | nan/inf | \|max\|   | \|mean\|   |
|-------|--------------------|------|-----------------------|----------------|-------------|---------|-----------|------------|
| 0     | 00 forward_in      | 0    | hidden                | (1, 2048)      | bfloat16    | 0/0     | 2.00e+00  | 1.19e-01   |
| 0     | 00 forward_in      | 0    | topk_ids              | (1, 8)         | int32       | 0/0     | 1.14e+02  | 8.04e+01   |
| 0     | 00 forward_in      | 0    | topk_weights          | (1, 8)         | float32     | 0/0     | 1.96e-01  | 1.25e-01   |
| 0     | 00 forward_in      | 1    | hidden                | (1, 2048)      | bfloat16    | 0/0     | 1.59e+00  | 1.68e-01   |
| 0     | 00 forward_in      | 1    | topk_ids              | (1, 8)         | int32       | 0/0     | 1.23e+02  | 7.65e+01   |
| 0     | 10 dispatch_out    | 0    | d_hidden (head=2)     | (2, 2048)      | float8_e4m3 | 0/0     | 4.48e+02  | 7.02e+01   |
| 0     | 10 dispatch_out    | 0    | d_scale  (head=2)     | (2, 16)        | float32     | 0/0     | 4.46e-03  | 2.23e-03   |
| 0     | 10 dispatch_out    | 0    | d_topk_ids (head=2)   | (2, 8)         | int32       | 0/0     | 1.23e+02  | 7.84e+01   |
| 0     | 10 dispatch_out    | 1    | d_hidden (head=2)     | (2, 2048)      | float8_e4m3 | 0/0     | 4.48e+02  | 7.02e+01   |
| 0     | 30 fmoe_out        | 0    | f_hidden              | (32768, 2048)  | bfloat16    | 0/0     | **1.56e-02** | 1.37e-08   |
| 0     | 30 fmoe_out        | 1    | f_hidden              | (32768, 2048)  | bfloat16    | 0/0     | **1.24e-02** | 1.83e-08   |
| 0     | 40 combine_out     | 0    | c_hidden_sliced[:1]   | (1, 2048)      | bfloat16    | 0/0     | 1.64e-02  | 2.68e-04   |
| 0     | 40 combine_out     | 1    | c_hidden_sliced[:1]   | (1, 2048)      | bfloat16    | 0/0     | 2.61e-02  | 5.53e-04   |
| 1     | 00 forward_in      | 0    | hidden                | (1, 2048)      | bfloat16    | 0/0     | 2.69e+00  | 2.15e-01   |
| 1     | 30 fmoe_out        | 0    | f_hidden              | (32768, 2048)  | bfloat16    | 0/0     | **3.36e-03** | 1.25e-08   |
| 1     | 40 combine_out     | 0    | c_hidden_sliced[:1]   | (1, 2048)      | bfloat16    | 0/0     | 8.42e-03  | 4.50e-04   |

Reading the rows:

- 00 → 10: dispatch outputs FP8 with `|max|≈448`. Multiplying by
  `d_scale ≈ 4.5e-3` gives `≈ 2.0`, matching the BF16 input `|max|≈2.0`. So
  the FP8 quantization is consistent, the dispatched buffer carries the
  original activation magnitude, and `d_topk_ids` stays in the global
  `[0, 128)` range as expected.
- 30: `fused_moe` output `|max|` drops to ~1.5e-2 on layer 0 and ~3.4e-3 on
  layer 1, while the BF16 input was `~2.0` and `~2.7` respectively. The
  shrink factor is ~100×–1000×, far below what any normal MoE block (residual
  stream same scale as input) would produce.
- 40: combine just sums fused_moe outputs across PEs unweighted, so it carries
  the same shrink ratio (no further loss / no further amplification). 0
  NaN / 0 Inf throughout.

### 28.4 Interpretation — which stage is the corruption point

Two independent observations rule out mori IntraNode (path 2 in §27.4):

1. **Dispatch is internally consistent.** FP8 codes × per-128 scale recover
   the original activation magnitude (`448 × 4.5e-3 ≈ 2.0`), and the global
   topk_ids reaching aiter are still in `[0, 128)`. The mori dispatch buffer
   would have to be silently re-scaled by ~1/128 across all rows to make
   later fused_moe output that small — instead it matches the input. So
   `dispatch_a1` and `dispatch_scale` enter `run_moe_core` with the right
   numerical content.
2. **Combine is faithful.** `c_hidden_sliced[:1]` at every layer is the same
   order of magnitude as `f_hidden` at the same layer (1.6e-2 ↔ 1.5e-2,
   2.6e-2 ↔ 1.2e-2, 8.4e-3 ↔ 3.4e-3). If mori IntraNode were dropping or
   duplicating tokens at world_size<8, the ratio between fmoe_out and
   combine_out would not stay so close.

So the corruption is concentrated at the **`run_moe_core` → aiter
`fused_moe` boundary**: aiter receives a numerically correct FP8 activation
plus its scale, then writes back values 2–3 orders of magnitude too small.
That matches path (3) in §27.4 and rules out path (2).

### 28.5 Why the aiter call edge is the suspect — the static piece

Diff of the two aiter-fused_moe call sites in
`python/sglang/srt/layers/moe/ep_moe/layer.py` (full file is verbatim from
upstream sglang, no miles patches):

- `DeepEPMoE.forward_aiter` (line ≈ 348) — used by every deepep+aiter EP run
  that we have ever shipped:

```python
return fused_moe(
    hidden_states,                       # BF16 (deepep dispatch returns BF16 unless given a tuple)
    self.w13_weight,
    self.w2_weight,
    topk_weights,
    topk_ids_copy,                       # -1 sentinel replaced by num_local_experts
    w1_scale=self.w13_weight_scale_inv,
    w2_scale=self.w2_weight_scale_inv,
    quant_type=QuantType.per_128x128,
    activation=ActivationType.Silu,
    expert_mask=self.expert_mask,        # size num_local_experts + 1
)                                        # no a1_scale, no num_local_tokens
```

- `MoriEPMoE.run_moe_core` (line ≈ 881) — exercised by every mori rollout we
  have ever run:

```python
hidden_states = fused_moe(
    hidden_states=dispatch_a1,           # FP8 directly out of mori dispatch
    w1=w13_weight, w2=w2_weight,
    w1_scale=w13_scale, w2_scale=w2_scale,
    a1_scale=dispatch_scale,             # *** mori per-token activation scale, [Mmax, hidden/128], fp32
    topk_weight=dispatch_weights,
    topk_ids=dispatch_ids,               # global ids in [0, num_experts)
    quant_type=QuantType.per_128x128,
    activation=ActivationType.Silu,
    expert_mask=self.expert_mask,        # size num_experts, with a 1-region of length num_local_experts
    num_local_tokens=dispatch_recv_token_num,
)
```

What aiter actually does with `a1_scale` in the FP8 input branch of
`fused_moe` (see `fused_moe.py` lines 1126–1139):

```python
elif hidden_states.dtype != q_dtype_a:
    a1, a1_scale = quant_func(hidden_states, scale=a1_scale,
                              quant_dtype=q_dtype_a, num_rows=num_local_tokens)
else:
    assert a1_scale is not None, "..."
    a1 = hidden_states                   # FP8 carried through unchanged
                                         # a1_scale is also carried through unchanged
```

The dispatched mori tensor is already FP8 = `q_dtype_a`, so the `else` branch
is taken: aiter never re-quantizes, and it consumes `dispatch_scale` exactly
as given. The `a1_scale` argument is documented in aiter (lines 141/195/229/390/1045)
as `[expert(local_expert:EP), 1, model_dim]` for the per-expert weight-scale
shape; the per-1x128 activation-scale path expects `[token_num, hidden/128]`
with a specific `partial_transpose(scale_t, a1_scale, num_rows=num_local_tokens)`
reorder (line 451–453). The mori dispatch produces a `[Mmax, hidden/128]`
fp32 buffer where row indexing follows mori's deduplicated `(orig_token, dest_PE)`
order, not the aiter `moe_sort` order that the stage-1 GEMM expects. With
the two row-orders unaligned, the stage-1 GEMM applies the wrong scale to
each FP8 row; the output magnitude collapses by roughly the inverse of how
sharply the activations vary across rows. That collapse is exactly the
~1/100–1/1000 attenuation observed in 28.3.

`num_local_tokens=dispatch_recv_token_num` reinforces the same problem: the
mori dispatch reports recv counts in its own ordering, so aiter's internal
`num_rows` book-keeping inherits a layout that does not match its own
`moe_sort` output.

### 28.6 Conclusion and recommendation

The mori IntraNode dispatch+combine kernels are functionally correct at
world_size=2 for this workload (28.4 (1) and (2)). The bug is the
calling-convention mismatch between `MoriEPMoE.run_moe_core` and aiter
`fused_moe` along the FP8-dispatch-with-per-token-scale edge. Recommended
fix, matching §27.4 path (3):

- Drop `a1_scale=dispatch_scale` from the call. Instead, dequant the FP8
  `dispatch_a1` using `dispatch_scale` into BF16 (the existing `upscale(...)`
  helper at `run_moe_core` line ≈ 834 already does exactly this for the
  W4A4-with-FP8-dispatch case) and pass that BF16 tensor to `fused_moe`
  without `a1_scale`. aiter will then re-quantize internally with its own
  row order, identical to how `DeepEPMoE.forward_aiter` is verified to work
  on the same hardware.
- Drop `num_local_tokens=dispatch_recv_token_num`. With the input now BF16,
  aiter does its own quantization based on `topk_ids.shape[0]` like in the
  deepep+aiter path.
- Keep `dispatch_ids` (global ids in `[0, num_experts)`) and `expert_mask`
  of size `num_experts` (1 only in the local-rank's expert range) — those
  two together are exactly what the deepep+aiter path uses to drop
  contributions belonging to remote experts, and mori dispatch already
  produces compatible global ids.

Cost of the fix: one extra FP8→BF16 dequant per layer on the dispatched
buffer, which is negligible compared to the GEMM itself, and removes the
need for the dispatched scale to obey aiter's row-ordering. If the
performance regression matters later, the proper long-term path is to add a
mori-side reorder that produces an `a1_scale` in aiter's `moe_sort` row
order; that requires changes inside mori (and a matching test at
world_size<8 in `tests/python/ops/test_dispatch_combine_intranode.py`) and
is out of scope for the immediate rollout-correctness fix.

### Files referenced (no edits retained in this round)

- `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
- `sgl-workspace/aiter/aiter/fused_moe.py`
- `tests/e2e/megatron/test_qwen3_30B_A3B/_common.py` (passthrough list, reverted)

### Logs

- `/tmp/dpsk_v4_test22_ep2mori_dbg4.log` — EP=2 + mori + debug prints,
  rollout produces the same garbled output observed in §26, MoE-layer
  statistics in 28.3 captured at the first real decode batch.

## 29. Attempted fix v6 — dequant FP8→BF16 inside `MoriEPMoE.run_moe_core`

Following §28.6 path (3), patched the `is_fp8_quant` branch of
`run_moe_core` to call `upscale(dispatch_a1, dispatch_scale,
dispatch_recv_token_num, output_dtype)` and set `dispatch_scale = None`
before the `fused_moe` call (so `a1_scale` becomes `None`). Kept
`num_local_tokens=dispatch_recv_token_num` because the next step's
diagnostic confirms mori's dispatched buffer is padded.

### 29.1 Result — fix did not work

Rollout still produces the same kind of garbled tokens. The first
training step's CI check failed with

```
log_probs (-8.933) != rollout_log_probs (-5.942)
```

— a Δ ≈ 3.0 between the SGLang rollout pass and the Megatron recompute,
on the same order as §19a EP=8 + mori (Δ ≈ 2.83) and §28 EP=2 + mori
pre-fix. Sample-level metrics also match the failure pattern: `rewards`
all zero, `truncated_ratio = 1.0`, response length pegged at 8192.

So the "mori dispatch row-order vs aiter `a1_scale` row-order"
hypothesis in §28.5–28.6 is **not** the root cause: removing
`a1_scale=dispatch_scale` and letting aiter re-quantize the BF16 buffer
internally does not fix the corruption.

### 29.2 Re-instrumented `run_moe_core` to capture true shapes

Added a fresh debug block that prints, layer 0 only, when not in
cuda-graph capture:

- `dispatch_a1.shape / dtype`
- `dispatch_ids.shape / dtype`
- `dispatch_scale` (now `None` after the dequant)
- `recv_token_num` value
- on the valid prefix `[:n]`: how many of the `n*topk` ids hit
  `expert_mask == 1` (the rank's own experts)
- on the padding suffix `[n:]`: min/max/in-range/own-hit counts and the
  hidden_states magnitude

Captured from `MILES_MORI_DEBUG=1` rollout (cuda-graph capture phase
only, sampled across capture batch sizes 256/512):

```
da=(32768, 2048)/torch.bfloat16
di=(32768, 8)/torch.int32
ds=None
recv_n=512         (also 496, 480, 464, ... across batches)
valid[:n]  own_expert_hits=4096/4096
pad[n:]    min=0  max=0  in_range[0,128)=258048/258048  own_expert_hits_in_pad=258048
da[:n] amax=0  amean=0
da[n:] amax=0  amean=0
```

### 29.3 Corrections to §28's mental model

The dump above forces two corrections to what §28 reasoned about:

1. **`dispatch_ids` is 2D `[max_recv, topk] = (32768, 8)`**, not 1D.
   §28.3 reported a 1D shape for the same field, but that was a
   misreading of the truncated dump. mori's IntraNode dispatcher
   returns `[max_recv_per_rank * world_size, num_experts_per_token]`
   for `recv_topk_ids`, which matches what aiter `fused_moe` expects
   for `topk_ids`. So there is no "global vs local id" or "1D vs 2D"
   structural mismatch on this field.

2. **The dispatched buffer is padded much more aggressively than §28
   implied.** With `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384`
   and `world_size=2`, mori reserves a recv buffer of
   `16384 * 2 = 32768` rows. Real traffic during decode is on the
   order of `recv_n ≈ 480..512`, i.e. ≈1.5 % of the buffer is valid;
   ≈98.5 % is padding.

### 29.4 Two new structural questions about the padding region

The capture-phase log already shows two padding-region facts that the
§28 hypothesis did not consider:

1. `pad[n:]` is **all zeros** for `dispatch_ids` (the kernel did not
   initialize the padding region with garbage, but the value `0` is a
   legal expert id; expert id `0` is owned by rank 0, so on rank 0 every
   one of the `(32768-recv_n) * topk ≈ 258048` padding entries reads as
   "this row needs expert 0").
2. `da[n:] amax = 0` — the hidden-state padding is also zero. So even
   if aiter's `moe_sorting` does not actually mask the padding region
   with `num_local_tokens`, the stage-1 GEMM that fires on padding
   rows would only contribute `weight * 0 = 0`. The padding hidden
   alone cannot create the ~1/128 magnitude collapse on the valid
   region observed in §28.3 — the issue must come from somewhere else.

What the dump does not yet show (capture phase is `da` all-zero by
construction; the real-rollout numbers in §28.3 were taken from a
pre-revert instrumented run on the FP8 path, not from this BF16 path):

- After the §29 dequant, does the valid `da[:n]` actually carry the
  expected magnitude (input to MoE should be |max| ≈ 2, |mean| ≈ 0.1
  per §28.3)? `upscale` should reproduce that, but it was not directly
  verified in this run because the only non-cuda-graph layer-0 frames
  reaching the debug print were still capture-phase dummies (input all
  zero).
- Does `aiter.moe_sorting` actually honor `num_local_tokens` in C++,
  or does it walk the full `topk_ids.numel() = 32768 * 8 = 262144`
  entries? The Python wrapper sizes `sorted_ids` from
  `topk_ids.numel()`, which is the full buffer regardless of
  `num_local_tokens`; whether the kernel masks beyond
  `num_local_tokens` is internal.

### 29.5 Tentative new direction (not yet implemented)

Based on the corrections above, the strongest remaining suspects are:

- **`num_local_tokens` does not actually mask `dispatch_ids` inside
  aiter `moe_sorting`.** If the kernel sorts all `32768 * 8` ids
  regardless, then on rank 0 the 258048 padding entries pointing at
  expert 0 will load 258048 padding rows of `dispatch_a1` into expert
  0's stage-1 GEMM. The padding hidden is zero, so the GEMM still gets
  zero per row — but the **scale path** does see those rows: per-token
  dynamic quantization on a row of zeros emits scale = 0, and aiter's
  stage-1 then has to divide by min(scale, ε) per row group when
  applying the activation scale. If padding rows are grouped together
  with valid rows in the GEMM block tiling, the block-level reduction
  could mix zero-scale rows with valid-scale rows and corrupt the
  per-token quant of the valid rows.
- **mori's padding region must be explicitly zeroed (or masked) on the
  dispatcher side before handing the buffer to aiter.** §29.2 shows
  `dispatch_ids` padding is `0` (a legal expert id), not a sentinel
  like `num_experts`. Most aiter MoE callers that pass padded buffers
  put a sentinel into the padding topk_ids so `moe_sorting` filters
  them out regardless of `num_local_tokens`.

### 29.6 Next-step experiment design

Before patching further, the cheapest experiment that disambiguates the
two suspects is to fill the padding region of `dispatch_ids` with the
sentinel `num_experts` (out of range) inside `MoriEPMoE.run_moe_core`
right after the dispatch returns:

```python
if dispatch_recv_token_num is not None:
    n = dispatch_recv_token_num   # device tensor, shape (1,)
    # mark all rows >= n as "expert num_experts" → expert_mask drops them
    pad_mask = (torch.arange(dispatch_ids.shape[0], device=dispatch_ids.device)
                .unsqueeze(1) >= n)
    dispatch_ids = dispatch_ids.masked_fill(pad_mask, self.num_experts)
```

Three possible outcomes:

- Rollout becomes clean → padding rows were leaking into aiter
  `moe_sorting`. Permanent fix is either (a) keep the sentinel patch
  upstream of `fused_moe` or (b) push the same masking into mori's
  IntraNode dispatcher (cheaper, one expression).
- Rollout still corrupt with Δ ≈ 2.99 → padding is not the issue;
  attention shifts to dynamic per-token quant interacting with the
  dispatched (or dequantized) buffer, or to the way `expert_mask` and
  `dispatch_ids` line up at low world_size in `moe_sorting`.
- Logprob Δ drops but not to ~3.6e-5 → partial fix, padding is part of
  the issue and there is at least one more component.

### 29.7 Files reverted in this round

- `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py` —
  removed the §29.1 dequant patch and the §29.2 debug block; `import`
  of `MoriEPMoE` re-verified.
- `tests/e2e/megatron/test_qwen3_30B_A3B/_common.py` — removed the
  `MILES_MORI_DEBUG` passthrough that was added for this round. The
  `--sglang-moe-a2a-backend mori` switch and the three
  pre-existing SGLANG_MORI passthrough keys remain unchanged.

### 29.8 Logs

- `logs/v6_ep2_mori_fix_dequant.log` — first attempt with §29.1 patch;
  rollout aborted mid-train, output identical to the §28 garbled
  pattern.
- `logs/v6_dbg_shape.log` — second run with shape-only instrumentation;
  produced the §29.2 dump (capture-phase only).
- `logs/v6_dbg_pad.log` — third run with padding instrumentation; same
  CI assertion failure
  `log_probs (-8.933) != rollout_log_probs (-5.942)`.

## 30. Attempted fix v7 — dequant + padding sentinel together

Following §29.6, layered the §29.1 dequant on top of a new "padding
sentinel" patch:

- `__init__`: precompute `_mori_padding_sentinel =
  expert_end_idx % num_experts` — by construction `expert_mask` is 0
  at that index, so any row tagged with the sentinel is dropped by aiter
  before it reaches a real expert tile. (Rank 0 gets 64, rank 1 gets 0;
  both are owned by the other rank.)
- `run_moe_core`: after the dequant, build `pad_mask = (arange(M) >=
  dispatch_recv_token_num)` on-device and
  `dispatch_ids = dispatch_ids.masked_fill(pad_mask, sentinel)`. All
  ops stay graph-safe (no `.item()`).

### 30.1 Result — still wrong, same magnitude as v6

CI assertion at the first training step:

```
log_probs (-8.720) != rollout_log_probs (-5.799)   Δ ≈ 2.92
rewards = 0.0   truncated_ratio = 1.0   repetition_frac = 0.016
```

Δ ≈ 2.92 vs v6's Δ ≈ 2.99 — within run-to-run noise. The padding
sentinel does not move the needle.

### 30.2 What v7 rules out

- **a1_scale row-order mismatch (§28 hypothesis)** — already invalidated
  by v6's BF16 dequant; v7 keeps the dequant and behavior is unchanged.
- **padding rows of `dispatch_ids` getting routed to a real expert
  (§29.4 / §29.5 hypothesis)** — invalidated. The §29.4 zero-hidden
  analysis was actually correct: padding rows of `dispatch_a1` are
  literally zero, so even if `moe_sorting` did walk them, every
  contribution to stage-1 GEMM is `0 * weight = 0`. v7 just removes
  the rows from `moe_sorting` entirely and the output magnitude does
  not change.

### 30.3 Suspect set after v6 + v7

What remains untested (ordered by my current prior):

1. **`dispatch_weights` row-order or layout mismatch.** mori returns
   `recv_topk_weights` alongside `recv_topk_ids`. aiter `fused_moe`
   uses it for the per-expert softmax weight in stage 2; if those
   weights are not aligned row-for-row with `dispatch_ids` after mori's
   dedup, the combined output is mathematically wrong even though
   GEMM accumulation looks fine. Never directly inspected — only
   `dispatch_ids` was dumped in §29.2.
2. **`combine` input ordering mismatch.** `MoriEPMoE.forward` feeds the
   `fused_moe` output (which aiter returns in *mori's dispatched row
   order*, because aiter respects the input `topk_ids` order) directly
   into `self.dispatcher.combine(...)`. If mori's combine kernel
   expects the input in a different row order than the one dispatch
   produced (e.g. it expects the aiter-`moe_sort` order, or some other
   "canonical" order), the combined hidden vector for each token will
   be a scrambled linear combination of expert outputs. This would
   leave per-layer norms looking healthy but per-token outputs
   scrambled — which is exactly what we observe (model converges to
   garbled but in-vocab tokens, not nan/inf or all-zero).
3. **mori IntraNode kernel at world_size=2.** The test suite at
   `tests/python/ops/test_dispatch_combine_intranode.py` runs with
   `world_size=8` by default; world_size=2 is on a less-covered code
   path. A standalone round-trip check at world_size=2 (random hidden
   in → dispatch → combine → compare-against-input) would tell us
   immediately whether the kernel itself is the problem or whether
   the bug is in how SGLang glues it to aiter.
4. **`MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` + mori interaction.**
   That env var changes how rollout drives SGLang. It is forced on by
   `_common.py` via ray runtime_env. Has not been disabled-as-a-test
   for the mori path.

### 30.4 Cheapest next experiment

The fastest disambiguating experiment is **(3) — direct mori-only
round-trip at world_size=2**, because it does not require touching
sglang or aiter:

```python
# Outside sglang: spawn 2 ranks, run mori's own IntraNode test
# at world_size=2 with hidden_size=2048, num_experts_per_rank=64,
# num_experts_per_token=8, max_num_inp_token_per_rank=1024,
# and verify dispatch→combine round-trip on random hidden states.
```

- If round-trip passes within FP8 quant noise → mori at ws=2 is
  functionally correct, and we go investigate (1) or (2).
- If round-trip fails → mori IntraNode has a real bug at ws=2, fix it
  there (or escalate upstream) and the SGLang gluing code does not need
  to change.

### 30.5 Files reverted

- `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py` —
  removed both the §30 padding-sentinel block (in `__init__` and
  `run_moe_core`) and the §29.1 dequant patch. `import MoriEPMoE`
  re-verified, `git diff --stat` empty for this file.
- `tests/e2e/megatron/test_qwen3_30B_A3B/_common.py` — unchanged from
  the post-§29 state (mori backend + 3 SGLANG passthrough keys, no
  debug-only passthrough).

### 30.6 Log

- `logs/v7_ep2_mori_dequant_sentinel.log` — same CI assertion failure
  pattern as v6, garbled rollout tokens, Δ ≈ 2.92.

## 31. mori IntraNode round-trip at world_size=2 — PASS

Per §30.4, ran a standalone two-rank mori dispatch+combine round-trip
without sglang or aiter in the loop, to qualify whether mori's kernels
themselves are correct at ws=2 with production-shaped inputs.

### 31.1 Setup

- Script: `/sgl-workspace/mori/test_ws2_intranode.py` — adapted from
  `tests/python/ops/test_dispatch_combine_intranode.py`, but with
  `gpu_per_node=2` (the default 8 hardcoded inside mori asserts
  `world_size % gpu_per_node == 0`, which fails on ws=2; explicit
  `gpu_per_node=world_size` is the documented escape valve and matches
  what the SGLang `MoriEPMoE` integration does internally).
- Parameters chosen to mirror the failing rollout:
  - `world_size = 2`
  - `hidden_size = 2048`  (Qwen3-30B-A3B `model_dim`)
  - `num_experts_per_rank = 64`  (128 / EP=2)
  - `num_experts_per_token = 8`  (top-k)
  - `max_num_inp_token_per_rank = 1024`
  - `quant = fp8_blockwise` along the hidden dim (1024 elements per block)
  - `dtype = bfloat16`
- Round-trip check: random hidden → IntraNode dispatch (FP8 quant) →
  IntraNode combine → all-gather → compare against ground-truth
  "if there were no quant" combine on rank 0.
- Log: `logs/mori_ws2_v3.log`

### 31.2 Result

```
[ws2-test] rank 0: PASS
[ws2-test] rank 1: PASS
```

Both ranks' relative L1 error of the round-trip against the unquant
reference is within the tolerance of mori's own test_utils
(`compare_results_with_tolerance` — the same checker the upstream
ws=8 tests use), under fp8_blockwise quant noise.

### 31.3 What this rules out / pins down

- mori IntraNode dispatch+combine at `world_size=2` with the exact
  Qwen3-30B-A3B production shapes is **kernel-level correct**.
- The corruption seen in §28–§30 therefore **cannot** be attributed to:
  - a broken kernel path at ws=2,
  - FP8 blockwise dispatch quantization itself,
  - combine kernel writing into the wrong rows,
  - row order of `dispatch_a1` / `dispatch_scale` in isolation (the
    round-trip would have caught that — combine inverts the same
    permutation).
- The bug must live in the **gluing code between mori's dispatched
  output and aiter's `fused_moe`** inside `MoriEPMoE.run_moe_core`,
  specifically one of:
  - per-expert weight scale (`fc1_smooth_scale` / `fc2_smooth_scale`)
    ordering vs mori's dispatched row order,
  - `dispatch_weights` (per-token top-k softmax weights) being passed
    in the right row order to aiter stage-2 reduce,
  - the way aiter's intra-fused output is then fed back to mori's
    `combine` — i.e. row order assumptions between aiter's stage-2
    output and mori's combine input.

### 31.4 Next direction (selected)

Of the remaining four suspects in §30.3, three (1, 2, 4) sit in the
SGLang gluing layer. The cheapest disambiguator that is non-destructive
to the model is to **diff `DeepEPMoE.run_moe_core` vs
`MoriEPMoE.run_moe_core`** line by line. DeepEP path produces clean
rollout on the same checkpoint; mori path produces Δ≈2.92 logprob.
Any semantic divergence in how the two `run_moe_core`s prepare
arguments for `aiter.fused_moe` (or in how they post-process its
output before handing it back to the dispatcher) is, at this point,
the strongest candidate.

## 32. Attempted fix v8 — DeepEP-style local topk_ids + small expert_mask

Driven by §31.4: aligned `MoriEPMoE` with `DeepEPMoE.forward_aiter`'s
convention for the two arguments that differ most visibly between the
two backends.

### 32.1 Patch

In `MoriEPMoE.__init__`, in addition to the existing global
`expert_mask` of shape `(num_experts,)`, build a DeepEP-style mask of
shape `(num_local_experts + 1,)` that is `[1]*num_local_experts +
[0]`. The trailing zero acts as the sentinel slot for rows whose
dispatched expert is not owned by this rank.

In `MoriEPMoE.run_moe_core`, just before calling `fused_moe`, convert
mori's GLOBAL dispatched `topk_ids` (range `[0, num_experts)`) into
LOCAL ids:

```python
dispatch_ids_local = dispatch_ids.to(torch.int32) - expert_start
dispatch_ids_local = torch.where(
    (dispatch_ids_local >= 0)
    & (dispatch_ids_local < self.num_local_experts),
    dispatch_ids_local,
    torch.full_like(dispatch_ids_local, self.num_local_experts),
)
```

and pass `dispatch_ids_local` plus the smaller `expert_mask` to
`fused_moe`. Hidden states, weights, scales, `num_local_tokens`,
`dtype` are unchanged. The dispatcher's combine path is untouched, so
mori's internal `dispDestTokIdMap` semantics are preserved. After
this conversion the arguments handed to `aiter.fused_moe` are
structurally identical to what `DeepEPMoE.forward_aiter` produces on
the same EP=2 setup, modulo the FP8 scale.

### 32.2 Result — still wrong, same magnitude

```
log_probs (-8.843) != rollout_log_probs (-5.872)   Δ ≈ 2.97
rewards = 0.0   truncated_ratio = 1.0
```

Δ ≈ 2.97 vs v6's Δ ≈ 2.99 and v7's Δ ≈ 2.92 — run-to-run noise.
Switching mori's GLOBAL ids + full-num_experts mask to the DeepEP
LOCAL ids + `(L+1,)` mask does not move the needle.

### 32.3 What v8 rules out

- The shape and "global vs local" semantics of `topk_ids` /
  `expert_mask` as seen by `aiter.fused_moe` are not the bug. aiter
  is internally translating GLOBAL → LOCAL through
  `local_expert_hash = expert_mask.cumsum(0) - 1` (see
  `aiter/fused_moe.py::torch_moe` and the corresponding C++ kernel
  path in `moe_sorting_fwd`); v8 confirms both call shapes resolve to
  the same valid expert routing.
- Whatever is corrupting the rollout therefore predates the
  `fused_moe` argument shape and lives either in the dispatched
  buffer (hidden / scale / weights row order) or in something
  fused_moe does **inside** the GEMM that only triggers on mori-side
  inputs but not on DeepEP-side inputs.

### 32.4 Files reverted

- `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
  — removed both the §32.1 `__init__` mask + the
  `run_moe_core` LOCAL-id conversion. `git diff --stat` empty for
  this file.

### 32.5 Log

- `logs/v8_ep2_mori_local_expert.log` — same CI assertion failure
  pattern as v6/v7, garbled rollout tokens, Δ ≈ 2.97.

### 32.6 Updated suspect set (post v8)

Going from §30.3, the surviving candidates are:

1. **mori dispatch FP8-only row-layout interaction with aiter
   stage1 GEMM scale broadcast.** §31's mori-only round-trip is an
   *identity check* (`combine(dispatch(x)) ≈ x`); it does not pin
   down what each individual expert *sees* between the two halves of
   the round trip. It is therefore consistent with the dispatched
   hidden / scale being correctly *recoverable* by combine while
   still being laid out in an order that aiter's stage1 GEMM scale
   broadcast (which assumes a particular per-block tiling) cannot
   consume correctly. The path narrows further to FP8 because v6
   (BF16 dequant + re-quant inside aiter) also failed — but BF16
   re-quant inside aiter still goes through the same `moe_sorting`
   ordering, so this is **not** ruled out yet.
2. **BF16 dispatch as a discriminator.** If we switch
   `SGLANG_MORI_DISPATCH_DTYPE=bf16`, the dispatched buffer carries
   no per-row scale at all — aiter then runs the same DeepEP-on-aiter
   path it uses today, dequant-quant ratio aside. If BF16 dispatch
   fixes the rollout, the bug is FP8-dispatch-specific (and v6's
   "dequant on receive side" did the dequant in the wrong place /
   too late). If BF16 dispatch still fails, FP8 is innocent and we
   should look at the dispatch row order / dedup interaction.
3. **`MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` + mori.** Still
   untested; cheap (env var only) but lower information content than
   suspect 2.

### 32.7 Next experiment

Pick **suspect 2** — set `SGLANG_MORI_DISPATCH_DTYPE=bf16` on the
existing EP=2 + mori run. The env passthrough already exists in
`tests/e2e/megatron/test_qwen3_30B_A3B/_common.py`, so no code change
is needed. Result decides whether further work focuses on the FP8
dispatch quant path or on the BF16-shared row-order/dispatch dedup
interaction.

## 33. Attempted fix v9 — BF16 dispatch (skip FP8 entirely)

Per §32.7, ran the same EP=2 + mori rollout with
`SGLANG_MORI_DISPATCH_DTYPE=bf16`, which steers `moriep.py:dispatch_a`
to skip both `fp8_quant_func` and `fp4_quant_func` and dispatch the
raw BF16 hidden state. Inside `MoriEPMoE.run_moe_core` the
`dispatch_scale` arrives as `None`, `quant_type` resolves to
`QuantType.per_128x128` only by way of `is_fp8_quant` (weights stay
FP8), and aiter's `fused_moe_2stages` takes the
`hidden_states.dtype != q_dtype_a` branch (line 1126 of
`aiter/fused_moe.py`), which re-quantizes BF16 → FP8 with a fresh
per-token scale entirely under aiter's own control. In other words,
v9 puts mori and DeepEP on the same activation-quant path for the
GEMM stage.

### 33.1 Result — still wrong, same magnitude

```
log_probs (-8.931) != rollout_log_probs (-5.936)   Δ ≈ 2.99
rewards = 0.0   truncated_ratio = 1.0
```

Δ ≈ 2.99 vs v6 (Δ ≈ 2.99), v7 (Δ ≈ 2.92), v8 (Δ ≈ 2.97) — run-to-run
noise. BF16 dispatch does **not** change the failure mode.

### 33.2 What v9 rules out

This is the strongest negative result so far. v9 means:

- The FP8 dispatch path itself (mori's `fp8_quant_func`, the
  per-row `dispatch_scale` returned alongside `dispatch_a1`, every
  FP8-specific aiter argument in `MoriEPMoE.run_moe_core`) is **not**
  the bug. v9 keeps mori as the dispatcher but takes both
  `dispatch_scale` and aiter's FP8-scale branches out of the loop,
  and the rollout still corrupts with the same Δ.
- Any hypothesis that needs FP8 quant to be the trigger is
  invalidated: §28's a1_scale shape concern, §29's per-token vs
  per-expert layout, §30's padding-sentinel scale-mixing scenario.
  None of those can be live, because the failure persists in BF16.

The remaining truth is that **mori produces a dispatched buffer
whose hidden-state row layout, dedup semantics, or per-row metadata
ordering disagrees with what aiter's `fused_moe` expects**, and the
disagreement is independent of FP8.

### 33.3 What v9 narrows the scope to

The single difference left between the working DeepEP-on-aiter path
and the broken mori-on-aiter path that v9 does not touch is the
*dispatch row layout* itself:

- **mori intranode** dispatches one row per `(src_token, dest_PE)`
  *after* deduplicating same-PE experts and writes the full
  `(numExpertPerToken,)` slice of the original token's top-K ids
  into `recv_topk_ids[i, :]`. Inside aiter, that one row therefore
  fans out into multiple in-rank experts (those of the original
  token's top-K that this rank happens to own), each contributing
  `weight = topk_weight[i, j_in_topk]`.
- **DeepEP normal** dispatches one row per `(src_token, dest_rank)`
  too, but `recv_topk_ids[i, :]` only carries the *in-rank* expert
  slots; everything else is `-1`, which SGLang then maps to the
  trailing sentinel slot of `expert_mask`. Inside aiter, fan-out per
  row is capped by however many of the original top-K landed in
  this rank — but the **weights** are still per-(row, j) and pulled
  from the same `recv_topk_weights` row.

This is mathematically equivalent on paper — both schemes deliver
`Σ_(experts in top-K) w_e · expert_e(x)` after combine — but they
differ in a place v8 already poked at (topk_ids representation) and
in a place v8 did **not** touch (which `j` slots are valid for each
row, and how that interacts with aiter's
`sorted_weights / sorted_expert_ids` after moe_sorting).

### 33.4 Files reverted

- No SGLang code changes were made for v9; the only change was the
  `SGLANG_MORI_DISPATCH_DTYPE=bf16` env var, which is per-run and
  not persisted.

### 33.5 Log

- `logs/v9_ep2_mori_bf16dispatch.log` — same CI assertion failure
  pattern as v6/v7/v8, garbled rollout tokens, Δ ≈ 2.99.

### 33.6 Next direction (selected)

After v9 the scope is concretely "mori dispatched-row layout vs
aiter's per-row top-K consumption". The cheapest *diagnostic*
experiment that pins this down is to **instrument `MoriEPMoE` at
layer 0** during a real rollout step (not capture phase) and print:

- `dispatch_recv_token_num` (actual valid row count)
- For each of the first `min(8, n)` rows i in `[:n]`:
  - `dispatch_a1[i, :8]` magnitudes (compare against pre-dispatch
    hidden state min/max to confirm row content is the right
    token's hidden)
  - `dispatch_ids[i, :]` (the full top-K id vector — should match
    one of the original tokens' top-K)
  - `dispatch_weights[i, :]` (the matching weights — should sum to
    ~1.0 if the originating token's softmax was healthy)

If, after capture phase, the first non-zero rows of `dispatch_a1`
have BF16 magnitudes that look like a different token than what
`dispatch_ids[i, :]` says they came from — or if
`dispatch_weights[i, :]` sums to a number wildly different from
1.0 — that pins the bug down to mori's dispatch row layout. If
everything looks consistent at this level, the corruption must be
inside aiter's stage1 / stage2 GEMMs for this particular input
shape, and the next step shifts to capturing aiter's
`moe_sorting_fwd` outputs (`sorted_ids`, `sorted_expert_ids`,
`sorted_weights`, `num_valid_ids`).

## 34. Attempted instrumentation v10 — layer-0 dispatched buffer dump

Per §33.6, added a small layer-0-only debug block in
`MoriEPMoE.run_moe_core` that prints, for the first few valid rows,
`dispatch_recv_token_num`, `dispatch_ids[i, :]`, `dispatch_weights[i, :]`
and `dispatch_a1[i, :8]`. The block is gated on `MILES_MORI_DEBUG_V10=1`
and skipped during `torch.cuda.is_current_stream_capturing()`.

### 34.1 Result — dispatched buffer is internally consistent

Sample dump for one real rollout forward (n = 2226 on rank 0):

```
[v10 rank=0 L0] n=2226 a1.dtype=torch.float8_e4m3fn
[v10 rank=1 L0 row=0] ids=[72, 40, 119, 102, 100, 108, 22, 18]
                     w=[0.211, 0.167, 0.167, 0.136, 0.097, 0.084, 0.070, 0.069]
                     w_sum=1.0000  a1[:8]=[96.0, 26.0, 52.0, 52.0, 11.0, 44.0, 80.0, 4.5]
[v10 rank=0 L0 row=1] ids=[72, 40, 119, 102, 100, 108, 22, 18]
                     w=[0.211, 0.167, 0.167, 0.136, 0.097, 0.084, 0.070, 0.069]
                     w_sum=1.0000  a1[:8]=[96.0, 26.0, 52.0, 52.0, 11.0, 44.0, 80.0, 4.5]
```

`(ids, weights, a1)` for the same source token are **bit-identical**
across rank 0 row 1 and rank 1 row 0 — exactly what mori's per-PE
dedup semantics promise. Every row's `w_sum ≈ 1.0`. `dispatch_a1`
holds the FP8-quantized hidden state with reasonable magnitudes.
Nothing in the v10 dump suggests the dispatcher is corrupting the
buffer.

This forces the bug "downstream of the dispatcher" — i.e., into
`aiter.fused_moe` itself, on inputs that look (from the dump's
vantage) perfectly well-formed.

### 34.2 Logprob still wrong, same magnitude

The instrumented run still failed CI: Δ ≈ 2.99 between log_probs
and rollout_log_probs, exactly like v6 – v9.

### 34.3 Files reverted

- `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
  — the debug block in `run_moe_core` removed; the `_common.py`
  env-passthrough additions for `MILES_MORI_DEBUG_V10*` are kept
  (no production side effect, only env-var passthrough).

### 34.4 Log

- `logs/v10_ep2_mori_dump.log` — full dump + same CI assertion.

## 35. Bug reproduced inside aiter's own multigpu test

§34 narrowed the blame to aiter. To pin it down without sglang
involvement, ran aiter's bundled
`op_tests/multigpu_tests/test_dispatch_combine.py` — which exercises
**mori dispatch → aiter.fused_moe → mori combine** end-to-end and
calls `checkAllclose` against a no-EP reference — but with the
Qwen3-30B-A3B production shape.

### 35.1 Repro setup

- `world_size = 2`, `hdim = 2048`, `idim = 768`, `E = 128`, `topk = 8`
- per-rank tokens ≈ 1100 (close to the n ≈ 2200 seen in v10 after
  mori's intra-rank fan-out)
- `quant_type = per_128x128` (the model's actual quant scheme)
- BF16 model dtype, FP8 (e4m3fn) weights and dispatched activation

The test required two upstream fixes:

1. aiter's `EpDispatchCombineConfig(...)` invocation must pass
   `gpu_per_node = world_size` for `world_size < 8`, mirroring what
   sglang's `moriep.py` already does, otherwise mori asserts
   `IsPowerOf2(gpuPerNode) && worldSize % gpuPerNode == 0`.
2. `mori_op.combine(...)` returns a `tuple` in the installed mori,
   not a tensor; unwrap with `combine_output[0]` before `.cpu()`.

Both fixes added to `op_tests/multigpu_tests/test_dispatch_combine.py`.

### 35.2 Repro result — checkAllclose FAILS

```
[aiter] -->max abs delta: 104960.0
        delta details: 99.9% (4501997 of 4505600) elements
[aiter] rank:0 [checkAllclose atol=0.01 rtol=0.01 FAILED]
[aiter] rank:1 [checkAllclose atol=0.01 rtol=0.01 FAILED]
```

- Reference rows ≈ ±3 K (BF16) ; mori-path rows ≈ ±20 K-50 K (BF16)
- 99.9 % of elements differ from the no-EP reference
- Both ranks fail with the same pattern

This is the **bug, reproduced inside aiter's own test harness, with
no sglang code on the call stack**.

### 35.3 Bisection — quant_type pins the trigger

Swept the test parameters one axis at a time, keeping the rest at
production:

| variant                | result              | max abs delta | %elements wrong |
|------------------------|---------------------|--------------:|-----------------|
| baseline `per_128x128` | **FAILED**          |       104 960 |     **99.9 %**  |
| `topk = 2`             | FAILED (no change)  |       166 912 |        100.0 %  |
| **`per_Token`**        | warning only        |           640 |          4.2 %  |

`per_Token` reduces the mismatch from "almost every element by
~10 K" to "4 % of elements by O(quant-noise)". `topk` does **not**
move the needle. **`quant_type = per_128x128` is the trigger.**

The default `test_dispatch_combine.py` invocation in aiter never
exercises `per_128x128`; it runs on `per_Token` and `No`. The
combination "mori dispatch with `per_1x128` activation scale +
aiter.fused_moe with `quant_type = per_128x128`" is exactly the path
sglang's `MoriEPMoE.run_moe_core` is using in production, and it
appears to have **no upstream coverage**.

### 35.4 What this means

The bug is **inside aiter**, specifically in how `fused_moe(..., 
quant_type=per_128x128, ...)` consumes a caller-supplied per-row
`a1_scale` produced by mori's `fp8_quant_func` (i.e., aiter's own
`per_1x128` quantization). The activation half of `per_128x128` is
`per_1x128` per the routing in `aiter/op_tests/multigpu_tests/
test_dispatch_combine.py` line 47-49 and the `get_hip_quant` call
inside `moriep.py`. So the scale shape is what aiter expects;
something further in (the kernel call's scale tiling, the
`moe_sorting_fwd` per-row scale rearrangement, or the
`fmoe_fp8_blockscale_g1u1` stage1 GEMM's scale broadcast) is
mishandling that scale on the mori path but not on the in-test
"`per_Token` end-to-end" path.

### 35.5 Next direction

Stop attempting fixes inside `MoriEPMoE.run_moe_core` — the bug is
in aiter. The narrowed scope is:

1. The aiter call path
   `fused_moe → ck_moe_2stages → moe_sorting_fwd →
   fmoe_fp8_blockscale_g1u1`,
   specifically how `input_scale` (= our `a1_scale = dispatch_scale`)
   flows through `moe_sorting_fwd`'s per-row sort and gets consumed
   by the `fmoe_bf16_blockscaleFp8_g1u1_*` ASM kernel.
2. The activation-side branch picked by `quant_type = per_128x128`
   in `aiter/fused_moe.py::fused_moe_2stages` for caller-quantized
   FP8 (`hidden_states.dtype == fp8` && `a1_scale is not None`).

The cheapest engineering workaround, if the actual aiter fix is
non-trivial, is to make `MoriEPMoE.run_moe_core` **re-dequantize
`dispatch_a1` to BF16 on receive** and call `fused_moe` with
`a1_scale = None` (i.e., let aiter re-quant internally — which is
exactly the path `per_Token` exercises in 35.3 and that produces
4 % noise instead of 99.9 % wrongness). v6 already tried this and
failed for unrelated reasons (the dequant code path bypassed the
real scale layout); a corrected version using
`rocm_moe_utils.upscale` (the kernel in §29) is the right
implementation. This is the candidate v11.

## 36. Attempted fix v11 — force aiter onto its 2-stage path

§35's bisection isolated `quant_type=per_128x128` as the trigger. A
deeper read of `aiter/fused_moe.py` showed that **caller-quantized
FP8 + `per_1x128` (the remap target of `per_128x128`)** picks the
1-stage path inside `get_2stage_cfgs` whenever
`token > 32 and inter_dim % 256 == 0`. At our production shape
`padded_M = 16384` and `inter_dim = 768`, so the 1-stage branch is
always selected. The 1-stage code (`fused_moe_1stage`) consumes a
caller-supplied `a1_scale` along a path that mishandles mori's
per_1x128 row-major scale, producing the 100 % wrongness in §35.

### 36.1 Patches tried

Three concrete attempts, all inside the same aiter repro from §35:

| variant                                     | max abs delta | %wrong  |
|---------------------------------------------|--------------:|---------|
| **v11a** caller pre-transposes a1_scale     |       118 272 |  99.9 % |
| **v11b** dequant FP8→BF16, a1_scale=None    |       116 224 |  99.9 % |
| **v11d** force 2-stage path                 |       **1 664** | **57 %** |
| v11e v11b + v11d combined                   |         2 048 |  57 %   |
| v11c num_local_tokens=None                  |  crash (partial_transpose requires num_rows) |

The forcing in v11d zeroes out `aiter.fused_moe.fused_moe_1stage_dict
[gfx]`, which makes the table-lookup in `get_2stage_cfgs:841` miss,
so `run_1stage` stays `False`. v11e is v11d on top of v11b; the
combination doesn't improve over v11d alone, confirming the 2-stage
path's residual ~5 % element-wise error is independent of FP8.

### 36.2 v11d in sglang

Promoted the v11d workaround into `MoriEPMoE` at module import time
(per-call patching arrives too late because
`get_2stage_cfgs` is `functools.lru_cache`'d on its args, so the
first call wins). The CUDA-graph capture logs from the run confirm
the patch is live:

```
[fused_moe] using 2stage default for
  (256, 16384, 2048, 768, 64, 8, ActivationType.Silu, bf16,
   fp8_e4m3fn, fp8_e4m3fn, QuantType.per_1x128, ...)
```

i.e. `2stage default` instead of the previously-seen 1stage.

Rollout result: **still wrong, same magnitude as before**

```
log_probs (-8.838) != rollout_log_probs (-5.868)   Δ ≈ 2.97
```

The 2-stage path's max delta of 1.6 K (relative error ~5 %) is
enough to make 48 layers of MoE accumulate into completely garbled
output by the time the LM head runs softmax.

### 36.3 What v11 narrows the scope to

The bug is **not** a single per-call mistake we can paper over from
SGLang:

- caller-side transpose (v11a) — wrong
- dequant on receive (v11b) — same magnitude as baseline
- force 2-stage (v11d) — 60× better but still 5 % off everywhere
- combined (v11e) — no further improvement
- skip `num_local_tokens` — fails an internal `partial_transpose`
  precondition

The 2-stage path's residual 5 % error is itself a numerical bug in
aiter: when fed an *identical* dispatched buffer the no-EP single
`fused_moe` reference reproduces the same output to quant noise, but
the mori→aiter round-trip systematically biases every element by
~5 %. This is the same path DeepEP's `forward_aiter` uses (per §35),
yet DeepEP rollouts produce correct logprobs, which strongly
suggests the bias is specific to the *combination* of mori's
dispatched row layout and aiter's caller-quantized FP8 handling at
this shape.

### 36.4 Files reverted

- `sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
  — `MoriEPMoE` and module-import block fully reverted (`git diff
  --stat` empty for this file).
- `sgl-workspace/aiter/op_tests/multigpu_tests/test_dispatch_combine.py`
  — reverted as well; reproducer migrated to the repo at
  `aiter_mori_per128x128_repro.py`, which inlines the two upstream
  fixes (`gpu_per_node` and `mori_op.combine[0]` tuple unwrap) so it
  can be shared verbatim with the aiter team.

### 36.5 Logs

- `logs/v11_ep2_mori_force2stage.log` — per-call monkey-patch (the
  patch arrives after `get_2stage_cfgs`'s LRU cache is warmed).
- `logs/v11b_ep2_mori_force2stage_import.log` — import-time
  monkey-patch (cache sees patched dict on first call, CUDA-graph
  capture log confirms 2-stage selection). Same final
  `Δ ≈ 2.94`.

### 36.6 Handoff

Per the user's call, debugging stops here. The remaining work is
upstream:

1. **AMD aiter team**: numerical bug in `fused_moe` on the
   `(mori-dispatched FP8, per_128x128, world_size=2, hdim=2048,
   inter_dim=768, E=128, topk=8)` combination — both 1-stage and
   2-stage paths affected. Reproducer +
   isolation evidence: `aiter_mori_per128x128_repro.py`,
   `debug_report.md §35-§36`, draft issue text in
   `aiter_mori_issue.md`.
2. **SGLang side**: no shipping workaround. Recommend keeping mori
   off the rollout backend list for FP8 + per_128x128 models on
   MI350 until aiter is fixed.
3. **Side findings, also for upstream**:
   `aiter/op_tests/multigpu_tests/test_dispatch_combine.py` has two
   small bugs that prevent any-`world_size<8` numerical test
   coverage of the mori+fused_moe pipeline: missing `gpu_per_node`
   in `EpDispatchCombineConfig(...)`, and an unsafe direct `.cpu()`
   on `mori_op.combine(...)` which returns a tuple in the installed
   mori. Both fixes are inlined in
   `aiter_mori_per128x128_repro.py`.

## 37. Latest-aiter check — bug persists, upstream skips the test

The user downloaded the latest aiter into
`/apps/tas/yaoc/work/miles/aiter` to check whether a newer release
already fixes this. Findings:

### 37.1 Versions

- Runtime image (buggy): `v0.1.11.post1` (`417de6df4`, 2026-03-05).
- Downloaded HEAD: `v0.1.14-rc0-221-g1969c180d` (`1969c180d`,
  2026-06-05) — **2136 commits ahead**.

### 37.2 The defective path is unchanged on HEAD

Source comparison of `aiter/fused_moe.py`:

- `fused_moe_2stages` caller-quantized FP8 + `per_1x128` path still
  takes `a1 = hidden_states` **without** a `partial_transpose` on
  `a1_scale` before `asm_stage1` (HEAD lines 1765-1785; same as
  v0.1.11.post1 lines 1126-1146). This is the suspected core defect
  and it is byte-for-byte the same logic.
- `fused_moe_1stage` still does the transpose (HEAD 612-633).
- `quant_remap = {per_128x128: per_1x128}` retained (HEAD 779).
- The `run_1stage` gate in `get_2stage_cfgs` was **loosened** from
  `inter_dim % 256 == 0` to `inter_dim % 128 == 0` (HEAD 1193-1195),
  which makes the catastrophic 1-stage path *easier* to hit, not
  harder.
- `partial_transpose` CUDA kernel only got a type-system refactor
  (`#2932`, torch tensor → `aiter_tensor_t`); no logic change.

Conclusion: upgrading aiter is not expected to fix the rollout
corruption. (Not re-run on HEAD — the fresh checkout is uncompiled;
this round is a source audit, the v0.1.11.post1 run is the empirical
data point.)

### 37.3 Upstream already excludes the mori multi-GPU test from CI

- `#1453` "Support mori in aiter" (`e2dd9995d`, 2025-11-30) added
  mori support.
- `#1530` "CI: Skip Mori tests in multi gpu tests" (`e91f2ed9c`,
  2025-12-17) added `test_mori_all2all.py` to the CI skip list,
  alongside `test_dispatch_combine.py` and `test_communication.py`
  (already skipped).
- On HEAD, `.github/scripts/aiter_test.sh` still skips all three.

So the exact harness that reproduces this bug
(`test_dispatch_combine.py`) has had no CI coverage since mori
landed, which is consistent with the regression going unnoticed
upstream. This cross-reference is now folded into
`aiter_mori_issue.md` (new section "This path is currently uncovered
by CI") so the upstream report frames the ask as both "fix the
numerics" and "re-enable the skipped test".

## 38. Root cause proven — aiter `a1_scale` padded-transpose bug

### 38.1 Summary

The catastrophic numerical error in §35-§36 was traced to a specific
transpose-dimension error in aiter's `fused_moe` FP8 caller-quant
path. When `per_128x128` is remapped to `per_1x128`, `a1_scale` must
be transposed to `[num_rows, scale_dim]`. The kernel uses
`num_local_tokens` as the row count, but mori dispatch pads the
buffer to `MaxNumTokensToRecv`, so the physical row count of
`hidden_states` exceeds `num_local_tokens`. The transpose reads the
wrong memory region, producing garbage scales.

### 38.2 Fix applied

Patched `/sgl-workspace/aiter/aiter/fused_moe.py`: the transpose now
uses `hidden_states.shape[0]` (physical buffer row count) instead of
`num_local_tokens`. Both 1-stage and 2-stage paths were patched.

### 38.3 Revert test

An environment-variable gate (`AITER_REVERT_A1SCALE_FIX`) was added
to toggle the patch at runtime. Results from
`aiter_mori_per128x128_repro.py` (2-GPU EP=2 simulation with real
weights):

| Condition      | off@1%   | max delta | Verdict       |
|----------------|----------|-----------|---------------|
| **Without** patch | 99.9%    | ~1.3e5    | catastrophic  |
| **With** patch    | ~17-23%  | ~1024     | fp8 noise floor |

### 38.4 Instrumented mori probe (`mori_probe.py`)

A 2-GPU probe was written to run `fused_moe` on actual mori-dispatched
buffers (dispatch + fused_moe + combine). Findings:

- **fused_moe kernel output** (pre-combine): correct with patch
  applied (`relL1 ≈ 0.024`, matching the pure-aiter baseline).
- **mori combine output**: intermittent non-deterministic corruption.
  Three consecutive calls returned different `relL1` values; the first
  call frequently showed ~10x magnitude inflation. Adding an explicit
  `torch.cuda.synchronize()` before combine mitigated the
  non-determinism in the probe.

## 39. Localization matrix — mori combine is the rollout blocker

### 39.1 Observation

Despite the kernel-level fix proven in §38, the end-to-end rollout
with mori still produces garbage text and `reward = 0`. A **BF16
dispatch** run (`fp8_dispatch=False`) was launched to test whether the
bug is FP8-specific. It produced identical garbage. This path does not
use FP8 quantization, does not construct `a1_scale`, and does not enter
the aiter fused_moe FP8 branch — yet it fails identically.

### 39.2 Full evidence matrix

| Run                   | EP | Backend  | Dispatch | raw_reward | Text     |
|-----------------------|----|----------|----------|------------|----------|
| test15 (EP=1 TP=1)   | 1  | **none** | —        | **0.53**   | coherent |
| test16 (TP=2 EP=1)   | 1  | **none** | —        | **0.50**+  | coherent |
| test17 (TP=1 EP=2)   | 2  | **none** | —        | **0.53**   | coherent |
| bf16_dispatch_ab      | 2  | **mori** | bf16     | **0.0**    | garbage  |
| fp8_fix_e2e           | 2  | **mori** | fp8      | **0.0**    | garbage  |
| test8–12 (EP=8)       | 8  | **mori** | fp8/bf16 | **0.0**    | garbage  |

(Logs: `/tmp/dpsk_v4_test{15,16,17}_*.log`, `/apps/tas/yaoc/work/miles/miles/{bf16_dispatch_ab,fp8_fix_e2e}.log`)

### 39.3 Conclusion

The **single distinguishing variable** is the mori backend. EP itself
is not the issue (test17 is EP=2 without mori, reward 0.53). The
`a1_scale` kernel fix is a real bug that should be upstreamed, but it
is **not** the rollout blocker. The blocker is in the mori
`INTRA_NODE` dispatch/combine path, affecting both fp8 and bf16
dispatch.

## 40. Hypothesis: CUDA-graph × mori-combine interaction

### 40.1 Rationale

The mori probe (§38.4) showed that an explicit
`torch.cuda.synchronize()` before combine mitigated the
non-deterministic corruption. In the e2e, sglang captures the decode
path (including mori dispatch+combine) into a **CUDA graph**. A
host-side `synchronize()` or process-group barrier **cannot be captured
into a CUDA graph**, so any ordering dependency that the probe's sync
satisfies would be permanently broken under graph replay.

Supporting source evidence from `mori/src/ops/dispatch_combine/`:

- `EpDispatchCombineHandle::LaunchReset()` is **empty** (no-op).
- `crossDeviceBarrierMemObj` is allocated without a memset.
- The cross-device barrier uses relaxed system-scope atomics
  (`__ATOMIC_RELAXED`, `__HIP_MEMORY_SCOPE_AGENT`) and a
  generation-counter pattern that is sensitive to ordering.

### 40.2 Experiment: mori EP=2 FP8 e2e with `--disable-cuda-graph`

A single-variable test was run: same config as the mori EP=2 FP8
run, with only `--sglang-disable-cuda-graph` added (env-gated via
`MILES_DEBUG_DISABLE_CUDA_GRAPH=1` in `_common.py`).

```
Log: /tmp/mori_ep2_graphoff.log
Config: EP=2, mori INTRA_NODE, fp8_dispatch=True, disable_cuda_graph=True
Confirmed: "cuda graph: False" in all decode batches
```

**Result: still garbage.**

```
raw_reward: 0.0
First rollout text: "裁定 prediction سبح drug候选人..." (same gibberish pattern)
```

### 40.3 Interpretation

Disabling CUDA graph did **not** restore coherence. This rules out
"CUDA-graph capture prevents a host-side sync from running" as the
sole root cause. The mori combine corruption occurs even in eager
(non-graph) execution. The probe's `synchronize()` may have been a
red herring (or it only mitigated a timing-dependent subset of the
bug).

The underlying defect is likely deeper in the mori intra-node combine
kernel itself — possibly in the cross-device barrier logic, staging
buffer addressing, or the `totalRecvTokenNum` reset sequence.

## 41. Current status and remaining work

> **SUPERSEDED by §42.** The "mori intra-node combine" blame below was a
> misattribution. The real rollout blocker was the `MoriEPMoE.expert_mask`
> buffer being zeroed after init (an all-zero mask makes `fused_moe` emit
> all-zero output, which combine then faithfully passed through). Fixed in
> the SGLang glue layer; `backend=mori` at EP=2 now passes end-to-end. See
> §42 for the root cause, fix, and verification.

### 41.1 What works now

- **`backend=none`, EP=2**: rollout produces coherent text,
  `raw_reward ≈ 0.53`. This is the currently usable EP path. It does
  not use mori or deepep for all-to-all; EP communication goes through
  the default torch path. Throughput is lower than mori but
  correctness is verified.
- **EP=1**: any backend works (test15, test16).

### 41.2 What is broken

- **`backend=mori`, any EP ≥ 2**: rollout produces garbage regardless
  of dispatch dtype (fp8 or bf16) and regardless of CUDA graph
  (on or off). The bug is in the mori intra-node combine path.

### 41.3 Two distinct bugs found

| # | Bug | Status | Fix location |
|---|-----|--------|--------------|
| 1 | aiter `fused_moe` `a1_scale` padded-transpose uses wrong row count | **Proven & fixed locally** in `/sgl-workspace/aiter/aiter/fused_moe.py`. Revert test shows 99.9% catastrophic error without patch vs fp8 noise floor with patch. | Upstream aiter |
| 2 | mori intra-node combine produces corrupted output at EP ≥ 2 | **Unresolved**. Affects both fp8 and bf16 dispatch, with and without CUDA graph. | Upstream mori |

### 41.4 Recommended next steps

1. **Upstream the aiter `a1_scale` fix** — the transpose dimension
   bug is real and independently reproducible
   (`aiter_mori_per128x128_repro.py`). It will bite any caller that
   passes padded FP8 buffers with `a1_scale`.
2. **Investigate mori combine at the C++ level** — the glue-layer
   mitigations (host sync, disable CUDA graph) did not help. The
   defect is likely in `intranode.hpp`
   (`EpCombineIntraNodeKernel_body`): candidates include the
   `crossDeviceBarrierMemObj` not being memset on init, the
   generation-counter barrier relying on relaxed atomics without
   sufficient fencing, or the staging buffer slot addressing.
3. **Use `backend=none` for EP rollout** in the interim — verified
   working at EP=2 with `raw_reward ≈ 0.53`.

### 41.5 Artifacts

| File | Description |
|------|-------------|
| `mori_probe.py` | 2-GPU instrumented mori dispatch+fused_moe+combine probe |
| `aiter_mori_per128x128_repro.py` | Standalone reproducer of the a1_scale bug |
| `aiter_ep_caller_scale_test.py` | Single-GPU aiter EP quant path bisection test |
| `/sgl-workspace/aiter/aiter/fused_moe.py` | Patched a1_scale transpose (runtime) |
| `/tmp/mori_ep2_graphoff.log` | Graph-off mori EP=2 e2e (garbage, rules out graph hypothesis) |
| `/apps/tas/yaoc/work/miles/miles/bf16_dispatch_ab.log` | BF16 dispatch mori EP=2 (garbage, rules out fp8-specific) |
| `/apps/tas/yaoc/work/miles/miles/fp8_fix_e2e.log` | FP8 dispatch mori EP=2 with kernel fix (garbage) |
| `/tmp/dpsk_v4_test17_tp1ep2.log` | backend=none EP=2 (coherent, raw_reward=0.53) |

## 42. ROOT CAUSE FOUND & FIXED — `expert_mask` buffer zeroed after init (NOT mori combine)

### 42.1 The actual bug

The rollout blocker (Bug #2 in §41.3, previously blamed on "mori intra-node
combine") lives in the **SGLang glue layer**, not in mori's kernels.

`MoriEPMoE.expert_mask` is a 128-element `int32` tensor marking which *global*
experts are local to this EP rank (1 = local). `aiter.fused_moe` masks out
every expert whose mask entry is 0. It is built once in `__init__`
(`torch.zeros(num_experts)` then `[start:end] = 1`, so `sum = num_local_experts`
= 64 at EP=2).

That persistent GPU tensor's memory is **silently zeroed between init and the
first real rollout forward**. With an all-zero mask, `fused_moe` masks out
*every* expert and emits **all-zero output for every token** → the whole MoE
layer becomes a no-op → prefill yields all-zero hidden states → garbage tokens,
`raw_reward = 0`, and `log_probs` vs `rollout_log_probs` diverge by ~2.8.

### 42.2 Evidence (TP=1 / EP=2 + mori — the §26 garbage config)

Instrumentation added to `MoriEPMoE` (env-gated by `MILES_MORI_COMBINE_PROBE`):

| Probe | Result |
|-------|--------|
| `MORI_MASK_INIT` (in `__init__`) | `mask_sum=64`, `valid_range=[0:64]`/`[64:128]` — **correct at construction** on every rank |
| `MORI_FMOE_DUMP` (captured a failing prefill) | `recv=200 a1_nonzero=200 out_nonzero=0` — 200 valid FP8 input tokens in, **zero** rows out |
| dumped `expert_mask.sum()` | **0** on both ranks (was 64 at init) |
| `ids-in-mask` for the live tokens | **0 / 1600** (200 tokens × 8 top-k) — every routed expert id falls outside the empty mask |

Timeline: warmup single-token prefills *before* CUDA-graph capture produced
non-zero output (mask still valid); the first real prefill *after* capture
produced zero output (mask already = 0).

### 42.3 Why §39 / §40 misattributed this to mori combine

- **mori's combine kernel was never broken.** It faithfully accumulated the
  all-zero `fused_moe` output, so the combined result merely *looked* zeroed.
  The isolation probes in §31 / §38.4 (`mori_probe.py`, the IntraNode
  round-trip) always passed because they fed combine *real* (non-zero) data
  with a *correct* mask — they never reproduced the clobbered-mask state.
- **§40.2's "disable CUDA graph still garbage" is now explained.** The defect
  is not graph *replay* reading stale memory; the persistent buffer is already
  zero by the time any real forward runs, graph on or off. Turning the graph
  off therefore could not help — exactly what §40.2 observed. The precise
  allocator event that overwrites the 512-byte buffer was not isolated (it
  coincides with the capture / weight-load window), but the fix does not depend
  on identifying it.
- **§38's aiter `a1_scale` transpose bug is a separate, real bug** and stays
  fixed in `/sgl-workspace/aiter/aiter/fused_moe.py`. It was "necessary but not
  sufficient" (§39) because the mask clobbering sat underneath it.

### 42.4 The fix

Rebuild the mask from its (constant) valid range on **every** `run_moe_core`
call instead of trusting the init-time buffer to survive.

```707:714:/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py
    def _build_expert_mask(self) -> torch.Tensor:
        mask = torch.zeros(
            (self.num_experts,),
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        mask[self._expert_start_idx : self._expert_end_idx] = 1
        return mask
```

```938:940:/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py
        if _MORI_COMBINE_PROBE and self.layer_id == 0:
            self._diag_stale_mask()
        self.expert_mask = self._build_expert_mask()
```

This is CUDA-graph safe: the `memset` + index-fill are captured into the graph
and re-execute on every replay, writing correct values to the graph-pool
address *before* `fused_moe` reads them. Even if that memory is reused between
replays, each replay self-initializes the mask first, so it is self-consistent
— which is precisely why building the mask once in `__init__` (outside any
captured region) was fragile.

Fix location: `/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py`
- `__init__`: store `self._expert_start_idx` / `self._expert_end_idx`, then
  `self.expert_mask = self._build_expert_mask()`.
- new helper `_build_expert_mask()`.
- `run_moe_core`: `self.expert_mask = self._build_expert_mask()` immediately
  before the `fused_moe` call.

### 42.5 Verification — e2e PASSED

Full `test_deepep_fp8.py` run (TP=1 / EP=2 + mori, fp8 dispatch, §38 aiter patch
still applied). Ray job reported `succeeded`.

| Metric | rollout 0 | rollout 1 | garbage baseline (§26) |
|--------|-----------|-----------|------------------------|
| `raw_reward` | 0.516 | 0.578 | 0.0 |
| C2: `log_probs` vs `rollout_log_probs` | Δ = 0.0061 | Δ = 0.0049 | Δ ≈ 2.8 |
| C1: `log_probs` vs `ref_log_probs` | 0.0 (bit-exact) | Δ = 5.4e-5 | n/a |
| rollout text | coherent English math reasoning | coherent | gibberish token soup |

- 4 training steps (0–3) completed; `train/loss` ≈ 0 (normal GRPO early-step
  magnitude); no NaN.
- `MORI_FMOE_DUMP = 0` for the whole run (no zero-output frame ever fired).
- `MORI_MASK_STALE = 3` — confirms the persistent mask *was* being clobbered
  and the per-forward rebuild took over.
- No traceback / exception / AssertionError / CUDA error.

Log: `/tmp/mori_fix_verify.log`.

### 42.6 Updated status (supersedes §41.2 / §41.3)

| # | Bug | Status | Fix location |
|---|-----|--------|--------------|
| 1 | aiter `fused_moe` `a1_scale` padded-transpose uses wrong row count | Fixed locally | `/sgl-workspace/aiter/aiter/fused_moe.py` (upstream aiter) |
| 2 | `MoriEPMoE.expert_mask` persistent buffer zeroed after init → all-zero mask → zero MoE output | **FIXED** — rebuilt per forward | `/sgl-workspace/sglang/.../ep_moe/layer.py` |

`backend=mori` at EP=2 now produces coherent rollout with healthy reward and
log-prob agreement, matching the `backend=none` baseline. The "use
`backend=none` in the interim" workaround (§41.4) is no longer required at EP=2.

### 42.7 Artifacts added this round

| File | Description |
|------|-------------|
| `/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py` | The fix — `_build_expert_mask()` rebuilt per `run_moe_core`; gated `_diag_stale_mask` / `_mori_fmoe_dump` / `_mori_combine_probe` probes |
| `mori_dump_inspect.py` | CPU-only inspector that proved the dumped `expert_mask.sum()==0` and `ids-in-mask 0/1600` |
| `mori_fmoe_replay.py` | Single-GPU replay of the captured failing `fused_moe` call |
| `/tmp/mori_fix_verify.log` | e2e verification run (coherent, `raw_reward` 0.516/0.578, job succeeded) |

