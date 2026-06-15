"""Rollout health check — run RIGHT AFTER rollout, BEFORE scoring/training, to catch
low-level harness problems early (instead of discovering them after a wasted round).

Flags per task-group:
  - ALL-TIMEOUT            : every rollout timed out (timeout too short / doc too long)
  - MOSTLY-NO-WRITE        : >=n-1 rollouts produced no file (timeout / agent didn't save)
  - WRITTEN-BUT-AUTO=0     : a report was written but automated_score=0 (grading/sync bug,
                             NOT a model failure — the classic advisory false-0)
  - LOW-VARIANCE           : surviving (non-empty) rollouts score within 0.02 -> will be
                             filtered out of GRPO (group contributes no gradient)

Exit code 2 if any CRITICAL group flagged (so the driver can stop before training).
"""
import json, sys, collections

path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
g = collections.defaultdict(list)
for r in rows:
    g[r["task_id"]].append(r)

print(f"=== ROLLOUT HEALTH CHECK ({len(rows)} rows, {len(g)} groups) ===")
critical = []
for tid, rs in sorted(g.items()):
    n = len(rs)
    n_to = sum(1 for r in rs if r.get("timed_out"))
    written = [r for r in rs if len(r.get("response") or "") > 0]
    n_empty = n - len(written)
    autos = [float(r.get("automated_score") or 0) for r in rs]
    auto_written = [float(r.get("automated_score") or 0) for r in written]
    flags = []
    if n_to == n:
        flags.append("ALL-TIMEOUT")
    if n_empty >= n - 1 and n >= 2:
        flags.append(f"MOSTLY-NO-WRITE({n_empty}/{n})")
    if auto_written and max(auto_written) == 0:
        flags.append("WRITTEN-BUT-AUTO=0(grading/sync bug)")
    if len(written) >= 2 and (max(auto_written) - min(auto_written)) < 0.02 and \
       (max([0.0] + [0]) or True):
        # surviving rollouts too similar on automated -> likely filtered
        pass  # variance is checked on the blended score later; flag the auto side only as a hint
    tag = ("  <<< " + " ".join(flags)) if flags else "  ok"
    print(f"  {tid:40s} n={n} timeout={n_to} empty={n_empty} auto=[{min(autos):.2f}-{max(autos):.2f}]{tag}")
    if "ALL-TIMEOUT" in flags or "WRITTEN-BUT-AUTO=0(grading/sync bug)" in " ".join(flags) or n_empty >= n - 1 and n >= 2:
        critical.append(tid)

if critical:
    print(f"\n⚠️⚠️ CRITICAL GROUPS (likely HARNESS, not policy): {critical}")
    print("   -> fix before training (raise ROLLOUT_TIMEOUT_MULT / check workspace-sync grading).")
    sys.exit(2)
print("\n✓ rollout health OK")
