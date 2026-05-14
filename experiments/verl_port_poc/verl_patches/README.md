# veRL Patches for Async Claw-Agentic RL

Patches required to make veRL's `experimental.fully_async_policy` actually run
for agentic RL with a claw runtime (jiuwenclaw / openclaw / any other). Without
these, training will deadlock on startup, NaN-crash on empty trajectories, or
silently train on zero-gradient batches.

**Base**: verl @ commit `8c3bee47` (`[tool] feat: simpler function-based tool registration (#6189)`)

**File**: `verl_async_for_claw_agentic_rl.patch` — unified diff covering 7
verl files. Apply with:

```bash
cd /path/to/verl
git apply /path/to/openclaw-verl-agent-rl/experiments/verl_port_poc/verl_patches/verl_async_for_claw_agentic_rl.patch
```

## What's in it (and why)

### 1. `experimental/fully_async_policy/message_queue.py`

`@ray.remote(num_cpus=2 → 0)`. Without this, the MessageQueue Ray actor stays
`PENDING` forever — the scheduler can't allocate 2 CPUs to it inside our pod's
constrained Ray cluster, training never starts. `num_cpus=0` lets the actor
run as a lightweight broker on whatever resources are available.

### 2. `experimental/fully_async_policy/detach_utils.py`

`param_version` None→0 sanitize. In STANDALONE vLLM mode `global_steps` is left
as `None` on some trajectory metadata, then `abs(None - None)` throws TypeError
mid-training. Patch coerces None to 0 before the subtraction.

### 3. `experimental/fully_async_policy/fully_async_rollouter.py`

Empty-trajectory filter (`[RolloutFilter]`). When the claw runtime times out
or returns nothing, the trajectory has `response_mask` all zeros. Trainer-side
mitigation alone (see #6) wastes a real training step on zero gradient. This
patch drops the trajectory in the rollouter, before it's pushed into the
MessageQueue, so trainer waits for a real sample instead. Adds
`count/dropped_stale_samples` accounting.

### 4. `experimental/fully_async_policy/fully_async_trainer.py`

Plumbing for the new filter counts; small debug print additions.

### 5. `experimental/separation/ray_trainer.py`

Two consecutive fixes in `_fit_compute_advantage`:

- **`[ZeroMaskFix]`** — defense-in-depth for empty trajectories that slip past
  the rollouter filter. Forces `response_mask[0]=1, token_level_rewards[0]=0`
  on all-zero rows so `compute_advantage` doesn't divide by zero and produce
  NaN that poisons the whole batch loss.

- **`[QualityFilter] race-to-bottom`** — when a GRPO group has
  `max_reward < 0.05`, zero out `token_level_rewards` for the group so its
  advantage = 0. Prevents the model imitating garbage samples that happened to
  "win" within a low-reward group. Includes a >50%-drops fallback that skips
  the filter if it would empty the batch (preserves *some* signal over noop).

### 6. `checkpoint_engine/base.py`

CheckpointEngine startup fix (gated by config). Required for FSDP2 separated
trainer/rollout topology. Without it, sleep/wake of vLLM replicas during
param sync misbehaves.

### 7. `workers/engine/fsdp/transformer_impl.py`

FSDP2 `sharded_save_to_cpu` workaround for LoRA-only models. Trainer was
asserting "No DTensor-type parameters" during ckpt save because the base
weights were fully offloaded and only LoRA adapters remained as DTensor.

## Validation checklist after applying

After applying these patches + setting `actor.strategy=fsdp2` +
`ref.strategy=fsdp2` + `algorithm.rollout_correction.bypass_mode=False` in
your launcher, you should see at startup:

- No `MessageQueue` actor `PENDING` (use `ray summary actors` to verify)
- `update_weights done, time cost: ~30-40s` after first param sync
- `[RolloutFilter] dropping empty trajectory ...` when claw runtime times out
- `[ZeroMaskFix] forced 1 valid token on N all-empty rows` (should be rare)
- `[QualityFilter] race-to-bottom: X/Y groups dropped` when reward is low
- No `TypeError` on `param_version_diff`
- No `AssertionError: No DTensor-type parameters` during ckpt save

## What's NOT in this patch

These remain as launcher-config knobs, not source patches:

- `algorithm.rollout_correction.bypass_mode=False` — bypass mode silently
  drops `rollout_log_probs` and crashes with ValueError if your batch needs
  them. Always set to False unless you know what you're doing.
- `actor.strategy=fsdp2 ref.strategy=fsdp2` — FSDP1 doesn't work with separated
  trainer/rollout topology.
- `async_training.staleness_threshold=0.3` — tune based on rollout/train ratio.
- `actor_rollout_ref.rollout.checkpoint_engine.backend=nccl` — bucketed NCCL
  transfer is the only reliable backend for disaggregated trainer/rollout.

See `experiments/verl_port_poc/launch_meeting_jiuwen_async.sh` for a complete
working launcher (with jiuwenclaw-specific bits) as reference.
