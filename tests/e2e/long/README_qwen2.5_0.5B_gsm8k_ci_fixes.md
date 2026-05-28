# Qwen2.5 0.5B GSM8K long CI fixes (sync + async)

Failures and fixes for running these e2e tests on **ROCm / gfx950** with **Python 3.10**. Both tests use `--megatron-to-hf-mode bridge` and the GSM8K CI metric checker.

| Test | Entry script | Est. time | Notes |
|------|----------------|-----------|--------|
| `test_qwen2.5_0.5B_gsm8k.py` | `train.py` | 6000s | Colocated (`--colocate`) |
| `test_qwen2.5_0.5B_gsm8k_async.py` | `train_async.py` | 5000s | Disaggregated actor/rollout GPUs |

---

## Issue 1: `StrEnum` import on Python 3.10

### Error

```
ImportError: cannot import name 'StrEnum' from 'enum' (/usr/lib/python3.10/enum.py)
```

### Solution

`miles/utils/enum_compat.py` backports `StrEnum` on 3.10; import sites use that module.

### Files modified

| File | Change |
|------|--------|
| `miles/utils/enum_compat.py` | **New** |
| `miles/utils/chat_template_utils/tito_tokenizer.py` | `from miles.utils.enum_compat import StrEnum` |
| `miles/utils/test_utils/session_verify_agent.py` | Same |

---

## Issue 2: `ReloadableProcessGroup` not pickleable (bridge `update_weights`)

### Error

```
TypeError: cannot pickle 'ReloadableProcessGroup' object
```

### Solution

Shim in `miles_plugins/megatron_bridge/__init__.py` strips `ReloadableProcessGroup` in `remove_non_pickleables`.

### Files modified

| File | Change |
|------|--------|
| `miles_plugins/megatron_bridge/__init__.py` | `_install_remove_non_pickleables_reloadable_pg_shim()` |

---

## Issue 3: TE wgrad — “Unable to find any suitable algorithms” (ROCm / gfx950)

### Error

```
RuntimeError: Unable to find any suitable algorithms
```

### Solution

1. `model_provider.py`: forward `gradient_accumulation_fusion` from CLI when not `None`.
2. Tests (ROCm only): `--no-gradient-accumulation-fusion`, `--ci-disable-logprobs-checker`.

### References

- Sync: [radixark/miles#1159](https://github.com/radixark/miles/pull/1159)
- Async: [radixark/miles#1160](https://github.com/radixark/miles/pull/1160)

### Files modified

| File | Change |
|------|--------|
| `miles/backends/megatron_utils/model_provider.py` | Honor `gradient_accumulation_fusion` before `finalize()` |
| `tests/e2e/long/test_qwen2.5_0.5B_gsm8k.py` | ROCm CLI flags; PR #1159 in comments |
| `tests/e2e/long/test_qwen2.5_0.5B_gsm8k_async.py` | ROCm CLI flags; PR #1160 in comments |

---

## Issue 4: Ray SIGSEGV at job teardown (ROCm)

### Error

SIGSEGV in `TaskInfoAccessor::AsyncAddTaskEventData` during actor kill (often after `[MetricChecker] pass dispose check`).

### Solution

`RAY_enable_task_events=0` on ROCm via `rocm_ray_runtime_env_vars()` in ray utils, rollout engines, train actors, and test `extra_env_vars`.

### Files modified

| File | Change |
|------|--------|
| `miles/ray/utils.py` | `rocm_ray_runtime_env_vars()` |
| `miles/ray/rollout.py` | SGLangEngine + Lock actors |
| `miles/ray/actor_group.py` | Train actors |
| Both test files | `**rocm_ray_runtime_env_vars()` in `extra_env_vars` |

---

## ROCm detection (both tests)

```python
IS_ROCM = getattr(torch.version, "hip", None) is not None
```

CUDA runs are unchanged.

---

## References

- [radixark/miles#1159](https://github.com/radixark/miles/pull/1159) — Sync test gradient accumulation fusion
- [radixark/miles#1160](https://github.com/radixark/miles/pull/1160) — Async test gradient accumulation fusion
- [ray-project/ray#51527](https://github.com/ray-project/ray/issues/51527) — Task event buffer SIGSEGV on disconnect
- [ray-project/ray#52374](https://github.com/ray-project/ray/pull/52374) — Move task-event flush to `Shutdown`
