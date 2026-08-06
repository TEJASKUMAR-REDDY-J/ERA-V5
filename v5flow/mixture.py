"""Session 5's plan, compiled into per-step quotas.

The plan is prose ("Indic 16%, floor 12%, warm the transitions"). The trainer
needs an integer: at step 137, how many of the 8 sequences in this microbatch
must come from the Indic lane. This module does that conversion, including
warmup blending between stages and enforcement of the protected floors.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import config


@dataclass
class StepPlan:
    step: int
    stage: str
    is_anneal: bool
    sequence_length: int
    mixture: dict[str, float]          # effective shares after warmup blending
    quota: dict[str, int]              # integer sequences per lane, sums to n_seq
    floors_applied: list[str]          # lanes lifted by a protected floor
    warmup_alpha: float                # 0 = fully previous stage, 1 = fully current


def stage_for(step: int) -> dict:
    for s in config.STAGES:
        if s["step_start"] <= step < s["step_end"]:
            return s
    return config.STAGES[-1]


def _blend(prev: dict, cur: dict, alpha: float) -> dict:
    lanes = set(prev) | set(cur)
    return {l: (1 - alpha) * prev.get(l, 0.0) + alpha * cur.get(l, 0.0) for l in lanes}


def effective_mixture(step: int) -> tuple[dict[str, float], dict, float]:
    """Blend across the stage boundary. V4 changed the Hindi share in one hard
    step against frozen embeddings and the gradient norm jumped ~150x, so every
    transition is spread over a warmup band instead."""
    st = stage_for(step)
    idx = config.STAGES.index(st)
    warm = st.get("warmup_steps", 0)
    since = step - st["step_start"]
    if idx > 0 and warm > 0 and since < warm:
        alpha = (since + 1) / warm
        return _blend(config.STAGES[idx - 1]["mixture"], st["mixture"], alpha), st, alpha
    return dict(st["mixture"]), st, 1.0


def apply_floors(mix: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Lift any lane below its protected floor, then renormalise the unprotected
    lanes so the shares still sum to one."""
    out = dict(mix)
    lifted = []
    for lane, floor in config.PROTECTED_FLOORS.items():
        if out.get(lane, 0.0) < floor:
            out[lane] = floor
            lifted.append(lane)
    if not lifted:
        return out, []
    protected = set(config.PROTECTED_FLOORS)
    fixed = sum(out[l] for l in protected if l in out)
    free = {l: v for l, v in out.items() if l not in protected}
    room = max(0.0, 1.0 - fixed)
    tot = sum(free.values()) or 1.0
    for l in free:
        out[l] = free[l] / tot * room
    return out, lifted


def shares_at(step: int) -> tuple[dict[str, float], dict, float, list[str]]:
    mix, st, alpha = effective_mixture(step)
    mix, lifted = apply_floors(mix)
    total = sum(mix.values()) or 1.0
    return {l: mix.get(l, 0.0) / total for l in config.LANES}, st, alpha, lifted


_QUOTA_CACHE: dict[int, list[dict[str, int]]] = {}


def quota_timeline(n_seq: int) -> list[dict[str, int]]:
    """Integer quotas for every step, carrying the rounding deficit forward.

    A lane at 4% of an 8-sequence microbatch rounds to zero every single step,
    which silently empties a protected lane. Tracking the running deficit means
    that lane is served every ~3 steps instead of never, and the realised shares
    converge on the plan.

    This is a pure function of the config -- no run state -- so every process
    and every replay computes the identical table.
    """
    if n_seq in _QUOTA_CACHE:
        return _QUOTA_CACHE[n_seq]
    table: list[dict[str, int]] = []
    deficit = {l: 0.0 for l in config.LANES}
    for step in range(config.TOTAL_STEPS):
        shares, _, _, _ = shares_at(step)
        want = {l: deficit[l] + shares[l] * n_seq for l in config.LANES}
        base = {l: int(want[l]) for l in config.LANES}
        rem = n_seq - sum(base.values())
        order = sorted(config.LANES, key=lambda l: (want[l] - base[l]), reverse=True)
        for l in order[:max(0, rem)]:
            base[l] += 1
        for l in config.LANES:
            deficit[l] = want[l] - base[l]
        table.append(base)
    _QUOTA_CACHE[n_seq] = table
    return table


def compile_step(step: int, n_seq: int) -> StepPlan:
    shares, st, alpha, lifted = shares_at(step)
    quota = quota_timeline(n_seq)[min(step, config.TOTAL_STEPS - 1)]
    return StepPlan(step=step, stage=st["stage"], is_anneal=bool(st.get("anneal")),
                    sequence_length=st["sequence_length"], mixture=shares,
                    quota=dict(quota), floors_applied=lifted, warmup_alpha=alpha)


def compile_timeline(n_seq: int) -> list[StepPlan]:
    return [compile_step(s, n_seq) for s in range(config.TOTAL_STEPS)]


def feasibility(registry, log) -> dict:
    """Compare planned lane demand against the tokens that actually exist.
    A lane that cannot be served from its shards has to repeat, synthesise or
    shrink -- saying so here is the whole point of the compiler."""
    plan = compile_timeline(config.MICRO_BATCH * config.GRAD_ACCUM)
    demand = {l: 0 for l in config.LANES}
    for p in plan:
        for lane, q in p.quota.items():
            demand[lane] += q * p.sequence_length
    supply = {l: 0 for l in config.LANES}
    for sid, m in registry.manifests.items():
        if m["lane"] in supply and registry.admit(sid)[0]:
            supply[m["lane"]] += m["token_count"]
    report = {}
    for l in config.LANES:
        d, s = demand[l], supply[l]
        epochs = d / s if s else float("inf")
        verdict = ("covered" if d <= s else
                   "needs_repetition" if d <= 4 * s else "supply_limited_synthesize")
        report[l] = {"demand_tokens": d, "supply_tokens": s,
                     "epochs": round(epochs, 3), "verdict": verdict}
        log.event("lane_feasibility", lane=l, demand=d, supply=s,
                  epochs=round(epochs, 3), verdict=verdict)
    return report
