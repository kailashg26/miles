# Fix: Qwen3-30B-A3B-FP8 RL rollout garbage on the SGLang + mori + aiter EP path

End-to-end RL training of **Qwen3-30B-A3B-FP8** on AMD MI350/MI355X produced
**garbage rollout text** and `raw_reward = 0` whenever the SGLang rollout used the
**mori** expert-parallel all-to-all backend at `EP ≥ 2`. The CI gate
(`log_probs` vs `rollout_log_probs`) diverged by ≈ 2.8.

This document describes the root causes, the fix, and the verification. The full
investigation log lives in [`../debug_report.md`](../debug_report.md) (§38 and
§42 are the relevant conclusions).

---

## 1. Scope — which repos change

The fix touches **three** repositories. **mori is NOT modified** — its
dispatch/combine kernels were never the bug.

| Repo  | Path                       | Branch / base commit                                                            | Patch                          | Role |
|-------|----------------------------|---------------------------------------------------------------------------------|--------------------------------|------|
| aiter | `/sgl-workspace/aiter`     | detached `417de6df4392120b766537efd50c7725cfa0d5af`                             | [`aiter.patch`](./aiter.patch) | FP8 block-scale kernel fix |
| sglang| `/sgl-workspace/sglang`    | detached `a929eb72882a8c47478de4d1eefd4a8ebb0716ff` (`[sglang-miles] Cherry-pick #24851`) | [`sglang.patch`](./sglang.patch) | `expert_mask` rebuild + mori routing |
| miles | this repo (your checkout)  | `main` @ `067ebff6e4afb86e47d9fbcf845499709163789a`                       | [`miles.patch`](./miles.patch) | env shim + test harness |
| mori  | `/sgl-workspace/mori`      | detached `e94694c79ad1ff78ca8cfeefccad0bcda2a24c95`                             | — (unchanged)                  | none |

---

## 2. Symptom

- Rollout text is incoherent (Chinese / programming-symbol token soup unrelated
  to the English math prompt).
- `rollout/raw_reward = 0`.
- CI gate C2 (`log_probs` vs `rollout_log_probs`) Δ ≈ 2.8 (healthy is ≈ 0.005).
- Only on `backend=mori`, `EP ≥ 2`. `backend=none` at the same EP is coherent.

---

## 3. Root causes

Two independent real bugs, plus environment enablers needed to run the test on
the MI350 image.

### Bug A — aiter `fused_moe` `a1_scale` padded-transpose uses the wrong row count

*(repo: aiter, file: `aiter/fused_moe.py`, debug_report §38)*

The asm block-scale stage-1 kernels read the per-`1x128` activation scale in a
column-major `[scaleN, token_cnt]` layout whose leading dimension is
`token_cnt = hidden_states.shape[0]` (the **physical** row count). The previous
code transposed the scale with `aiter.partial_transpose(num_rows=num_local_tokens)`,
i.e. using the **valid-token** count as the leading dimension. When the caller
passes an already-FP8 activation buffer **padded to a max capacity** (exactly
what a mori EP dispatch buffer is), `num_local_tokens < shape[0]`, so every
scale group `g > 0` is read at the wrong offset → catastrophic numerical error.
The 2-stage path was missing the transpose entirely on the caller-quantized-FP8
branch.

This was independently reproducible (`../aiter_mori_per128x128_repro.py`): 99.9%
catastrophic error without the patch vs the FP8 noise floor with it.

### Bug B — SGLang `MoriEPMoE.expert_mask` is zeroed after init (the rollout blocker)

*(repo: sglang, file: `python/sglang/srt/layers/moe/ep_moe/layer.py`, debug_report §42)*

`expert_mask` is a 128-element `int32` tensor that marks which **global** experts
are local to this EP rank (`1` = local). `aiter.fused_moe` masks out every expert
whose mask entry is `0`. It was built **once** in `__init__`
(`torch.zeros(num_experts)` then `[start:end] = 1`, sum = `num_local_experts` = 64
at EP=2).

That persistent GPU tensor's memory is **silently zeroed between init and the
first real rollout forward** (it coincides with the CUDA-graph-capture /
weight-load window; the exact allocator event was not isolated). With an all-zero
mask, `fused_moe` masks out **every** expert and emits **all-zero output for every
token** → the whole MoE layer becomes a no-op → prefill produces all-zero hidden
states → garbage tokens, `reward = 0`, log-prob divergence ≈ 2.8.

Evidence (TP=1 / EP=2 + mori):

| Probe | Result |
|-------|--------|
| `MORI_MASK_INIT` (in `__init__`) | `mask_sum=64` — **correct at construction** on every rank |
| `MORI_FMOE_DUMP` (a failing prefill) | `recv=200 a1_nonzero=200 out_nonzero=0` — 200 valid FP8 tokens in, zero out |
| dumped `expert_mask.sum()` | **0** on both ranks |
| `ids-in-mask` for live tokens | **0 / 1600** — every routed expert id falls outside the empty mask |

**Why earlier analysis blamed mori combine:** the combine kernel faithfully
accumulated the all-zero `fused_moe` output, so its result merely *looked*
zeroed. Disabling CUDA graph did not help (debug_report §40.2) because the buffer
is already zero before any real forward runs, graph on or off. mori's kernels are
correct.

### Enablers (needed to run, not numerical bugs)

- **sglang `qwen3_moe.py`** — route the mori backend off `forward_normal` (so it
  uses the EP dispatch path / `MoriEPMoE.forward`). One line.
- **miles `enum_compat.py` + imports** — `enum.StrEnum` only exists on Python
  ≥ 3.11; miles targets 3.10 and the MI350 image ships 3.10. The shim re-exports
  the stdlib type on 3.11+ and backports it on 3.10. Without it the import fails
  before the test starts (debug_report §1).
- **miles `deepseek_v32.py` / `deepseek_v4.py`** — lazy-import `encoding_dsv32`
  so non-DeepSeek jobs stay importable on older sglang builds that don't export it.
- **miles test harness** (`_common.py`, `test_deepep_fp8.py`) — select the mori
  backend, pass the required env vars through Ray `runtime_env` to the SGLang
  subprocess, and set the EP=2 isolation config used for verification.

---

## 4. The fix (minimal core changes)

### aiter — transpose the full physical scale buffer

```python
def _transpose_blockscale_a1_scale(a1_scale):
    # leading dim must be the physical row count (hidden_states.shape[0]),
    # not num_local_tokens, so padded buffers read group g>0 at the right offset.
    return a1_scale.transpose(0, 1).contiguous()
```

Applied on both the 1-stage path (replacing
`partial_transpose(num_rows=num_local_tokens)`) and the 2-stage `asm_stage1`
caller-quantized-FP8 path (where it was missing). See `aiter.patch`.

### sglang — rebuild `expert_mask` on every forward

Before (built once, fragile):

```python
self.expert_mask = torch.zeros((self.num_experts), device=..., dtype=torch.int32)
expert_start_idx = self.moe_ep_rank * self.num_local_experts
expert_end_idx   = expert_start_idx + self.num_local_experts
self.expert_mask[expert_start_idx:expert_end_idx] = 1
```

After (remember the range, rebuild per `run_moe_core`):

```python
# __init__
self._expert_start_idx = self.moe_ep_rank * self.num_local_experts
self._expert_end_idx   = self._expert_start_idx + self.num_local_experts
self.expert_mask = self._build_expert_mask()

def _build_expert_mask(self) -> torch.Tensor:
    mask = torch.zeros((self.num_experts,), device=torch.cuda.current_device(), dtype=torch.int32)
    mask[self._expert_start_idx : self._expert_end_idx] = 1
    return mask

# run_moe_core, immediately before the fused_moe call:
self.expert_mask = self._build_expert_mask()
```

This is CUDA-graph safe: the `memset` + index-fill are captured into the graph
and re-execute on every replay, writing correct values to the graph-pool address
*before* `fused_moe` reads them. Even if that memory is reused between replays,
each replay self-initializes the mask first, so it is self-consistent — which is
exactly why building it once in `__init__` (outside any captured region) was
fragile.

> `sglang.patch` also contains **gated debug instrumentation**
> (`_mori_dbg`, `_mori_fmoe_dump`, `_mori_combine_probe`, `_diag_stale_mask`,
> `MORI_MASK_INIT`/`MORI_MASK_STALE`). It is all behind `MILES_MORI_SELFCHECK`
> and `MILES_MORI_COMBINE_PROBE` (default off) and has no effect on normal runs.
> The only unconditional change is the `expert_mask` rebuild.

---

## 5. Verification

Full `test_deepep_fp8.py` run: TP=1 / EP=2 + **mori**, fp8 dispatch, with all three
patches applied. Ray job reported `succeeded`.
(Log: `/tmp/mori_fix_verify.log`.)

| Metric | rollout 0 | rollout 1 | garbage baseline |
|--------|-----------|-----------|------------------|
| `raw_reward` | **0.516** | **0.578** | 0.0 |
| C2: `log_probs` vs `rollout_log_probs` | Δ = 0.0061 | Δ = 0.0049 | ≈ 2.8 |
| C1: `log_probs` vs `ref_log_probs` | 0.0 (bit-exact) | Δ = 5.4e-5 | n/a |
| rollout text | coherent English math reasoning | coherent | gibberish |

- 4 training steps (0–3) completed; `train/loss` ≈ 0 (normal GRPO early-step
  magnitude); no NaN.
- `MORI_FMOE_DUMP = 0` for the whole run (no zero-output frame ever fired).
- `MORI_MASK_STALE = 3` — confirms the persistent mask *was* being clobbered and
  the per-forward rebuild took over.
- No traceback / exception / AssertionError / CUDA error.

---

## 6. How to apply

Run from the root of your miles checkout (this repo). aiter and sglang live at
fixed paths in the image; the patches use `a/` `b/` repo-relative paths.

```bash
# aiter / sglang (image-fixed locations)
git -C /sgl-workspace/aiter  apply "$PWD/bug_fix/aiter.patch"
git -C /sgl-workspace/sglang apply "$PWD/bug_fix/sglang.patch"

# miles (this repo)
git apply bug_fix/miles.patch
```

aiter and sglang are editable installs, so the changes take effect without a
rebuild. Dry-run first with `git apply --check <patch>`.

Reproduce the test (from the miles repo root):

```bash
bash bug_fix/run_test.sh
```

[`run_test.sh`](./run_test.sh) sets the required env: `SGLANG_USE_AITER=1`,
`SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384`, `unset SGLANG_DEEPEP_BF16_DISPATCH`.
The `MILES_MORI_SELFCHECK` / `MILES_MORI_COMBINE_PROBE` debug switches are left
commented out (optional instrumentation, not needed for the fix).

---

## 7. Notes & caveats

- **Verified at EP=2.** `test_deepep_fp8.py` was set to `sglang_ep_size=2` for the
  isolation run; the original CI value was `8`. The `expert_mask` fix is
  EP-agnostic, but re-running at EP=8 is recommended before closing the CI item.
- **Peripheral tree changes NOT included in these patches** (present on the
  verified image but unrelated to this bug):
  - aiter: `ops/triton/attention/pa_mqa_logits.py` (`if False` gluon disable),
    `ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py` (`config = dict(config)`).
  - sglang: `pyproject.toml` / `pyproject_other.toml` / `pyproject_rocm.toml` and
    the untracked `sgl-kernel/csrc/**/*.hip` files (ROCm build/packaging).
- **aiter Bug A is upstream-worthy** — the padded-transpose row-count bug bites any
  caller passing padded FP8 buffers with `a1_scale`.
