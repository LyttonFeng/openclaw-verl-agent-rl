# Algorithm

The algorithm is a lightweight agent-task variant of REINFORCE++:

```text
OpenClaw multi-turn rollout
-> task16 turn-level process reward + terminal reward
-> task-specific EMA mean/variance normalization
-> token-span broadcast
-> REINFORCE++ update
-> no critic, no GAE, no masked_whiten
```

Key choices:

- `rollout.n=1`: each prompt runs one live agent episode.
- No critic / no GAE: task16 credit assignment is dominated by explicit workflow events and final artifact quality.
- `gamma=0.0`: no return-to-go propagation across turns.
- Turn reward: assistant turns get scalar process rewards from task16 event checks; terminal reward is added at the final turn.
- Task EMA normalization:

```text
(raw_reward - EMA_mean(task_id)) / sqrt(EMA_var(task_id) + 1.0)
```

The `+1.0` variance floor avoids sparse-reward advantage explosions.

The veRL REINFORCE++ path is patched so advantage is:

```text
advantages = returns * response_mask
```

instead of applying `masked_whiten` over all response tokens.
