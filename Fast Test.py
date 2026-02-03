#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperbolic Petri Net Simulator – Fast & Clean Version
=====================================================

Simulation of Petri nets with hyperbolic embedding visualization.
Tracks radius growth and hyperbolic distance from initial marking.

Features:
- Efficient: expensive hyperbolic computations only periodically
- Safe fingerprint (handles negative incidence values)
- Optional JSON + text output + matplotlib trajectory plot
- Natural termination detection

MIT License
"""

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class PetriNet:
    """Simple Petri net definition"""
    incidence: List[List[int]]
    initial_marking: List[int]

    @property
    def n_places(self) -> int:
        return len(self.initial_marking)

    @property
    def n_trans(self) -> int:
        return len(self.incidence[0]) if self.incidence else 0

    def can_fire(self, t: int, marking: List[int]) -> bool:
        return all(marking[p] + self.incidence[p][t] >= 0 for p in range(self.n_places))

    def fire(self, t: int, marking: List[int]) -> List[int]:
        return [marking[p] + self.incidence[p][t] for p in range(self.n_places)]


# ─── Hyperbolic embedding ───────────────────────────────────────────────────

def embed_marking(marking: List[int], radius_scale: float = 0.38, jitter: float = 0.008
                  ) -> Tuple[float, float]:
    """
    Embed marking into Poincaré disk.
    Angle = weighted average of place indices, radius ~ log(total tokens)
    """
    total = sum(marking)
    if total == 0:
        return 0.0, 0.0

    theta = 0.0
    weight_sum = 0.0
    n = len(marking)

    for i, cnt in enumerate(marking):
        if cnt == 0:
            continue
        frac = cnt / total
        angle = 2.0 * math.pi * i / max(1, n - 1)
        theta += frac * angle
        weight_sum += frac

    if weight_sum > 0:
        theta /= weight_sum

    r = math.tanh(radius_scale * math.log1p(total))
    if jitter != 0.0:
        r += random.uniform(-jitter, jitter) * (1.0 - r)

    r = max(0.0, min(0.999, r))
    return r * math.cos(theta), r * math.sin(theta)


def poincare_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Poincaré disk distance"""
    ax, ay = a
    bx, by = b
    aa = ax*ax + ay*ay
    bb = bx*bx + by*by
    diff2 = (ax - bx)**2 + (ay - by)**2
    denom = (1.0 - aa) * (1.0 - bb)
    if denom < 1e-10:
        return 999.0
    arg = 1.0 + 2.0 * diff2 / denom
    return math.acosh(max(arg, 1.0000001))


# ─── Admissibility (hyperbolic displacement bound) ──────────────────────────

def admissible(initial: List[int], current: List[int], steps: int,
               factor: float = 0.92) -> Tuple[bool, float, float]:
    """Check if total hyperbolic displacement is still within bound"""
    d = poincare_distance(embed_marking(initial), embed_marking(current))
    bound = factor * 3.4 * math.log1p(steps + 1)
    return d <= bound, d, bound


# ─── Visualization helpers ──────────────────────────────────────────────────

def ascii_bar(value: int, max_val: int, width: int = 22, char: str = "█") -> str:
    if max_val <= 0:
        return " " * width
    filled = int(width * min(value, max_val) / max_val)
    return char * filled + " " * (width - filled)


def print_marking_bars(marking: List[int], labels: Optional[List[str]] = None):
    if not marking:
        return
    max_tokens = max(marking) + 1
    lbls = labels or [f"p{i}" for i in range(len(marking))]
    print("  Marking:")
    for name, cnt in zip(lbls, marking):
        bar = ascii_bar(cnt, max_tokens)
        print(f"   {name:>3} | {bar} {cnt:2d}")


def print_trajectory_summary(radii: List[float], total_distances: List[float]):
    if not radii:
        print("  (no steps performed)")
        return

    print("  Trajectory summary (last up to 12 steps – d is last known):")
    n_r = len(radii)
    show_count = min(12, n_r)
    last_d = total_distances[-1] if total_distances else None

    for k in range(show_count):
        idx = n_r - show_count + k
        r = radii[idx]
        d_str = f"{last_d:6.3f}" if last_d is not None else "  —  "
        print(f" {idx+1:3d} | r = {r:5.3f}    total d ≈ {d_str}")


def optional_plot(steps: List[int], radii: List[float], distances: List[float], prefix: Optional[str] = None):
    if not MATPLOTLIB_AVAILABLE or not steps:
        return

    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    ax1.plot(steps, radii, "o-", lw=1.2, ms=4, label="radius", color="#1f77b4")
    ax1.set_ylabel("Disk radius")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if distances:
        # Plot only at steps where distance was actually computed
        dist_steps = steps[:len(distances)]
        ax2.plot(dist_steps, distances, "s--", lw=1.3, ms=5,
                 label="total hyp. dist", color="#d62728")
    ax2.set_ylabel("Hyperbolic distance from initial")

    fig.suptitle("Hyperbolic Petri Net Trajectory", fontsize=13)
    fig.legend(loc="upper center", ncol=2, fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    if prefix:
        path = f"{prefix}_trajectory.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")
    else:
        plt.show()


# ─── Fingerprint ────────────────────────────────────────────────────────────

def fingerprint(net: PetriNet, trace: List[int], final: List[int]) -> str:
    """Deterministic hash – safe for negative incidence values"""
    h = hashlib.sha3_256()

    # Incidence matrix (handle negatives via & 0xFF)
    for row in net.incidence:
        for val in row:
            h.update((val & 0xFF).to_bytes(1, byteorder='little'))

    # Trace (transition indices)
    for t in trace:
        h.update(t.to_bytes(2, byteorder='little', signed=False))

    # Final marking
    for cnt in final:
        h.update(cnt.to_bytes(4, byteorder='little', signed=False))

    return h.hexdigest()[:32]  # 128-bit prefix


# ─── Main simulation ────────────────────────────────────────────────────────

def run_simulation(
    net: PetriNet,
    max_steps: int = 800,
    output_prefix: Optional[str] = None,
    verbose: bool = True
):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_file = None
    json_path = None
    if output_prefix:
        out_file = open(f"{output_prefix}.txt", "w", encoding="utf-8")
        json_path = f"{output_prefix}.json"

    def log(msg: str = "", end="\n"):
        if verbose:
            print(msg, end=end)
        if out_file:
            out_file.write(msg + end)
            out_file.flush()

    log("Hyperbolic Petri Net Simulator")
    log(f"  started:      {ts}")
    log(f"  places:       {net.n_places}    transitions: {net.n_trans}")
    log(f"  initial:      {net.initial_marking}")
    log("─" * 66)

    marking = net.initial_marking[:]
    trace: List[int] = []
    radii: List[float] = []
    total_distances: List[float] = []
    step_numbers: List[int] = []           # steps when radius was recorded

    prev_total_tokens = sum(marking)
    last_full_check = 0

    for step in range(1, max_steps + 1):
        fired = False

        for t in range(net.n_trans):
            if net.can_fire(t, marking):
                marking = net.fire(t, marking)
                trace.append(t)
                fired = True

                curr_embed = embed_marking(marking)
                r = math.hypot(*curr_embed)
                radii.append(r)
                step_numbers.append(len(trace))

                curr_total = sum(marking)

                # Compute expensive hyperbolic distance & check admissibility periodically
                if (step - last_full_check >= 5
                        or curr_total != prev_total_tokens
                        or step == max_steps):
                    d = poincare_distance(embed_marking(net.initial_marking), curr_embed)
                    total_distances.append(d)

                    ok, _, bound = admissible(net.initial_marking, marking, len(trace))
                    if not ok:
                        log(f"\n!!! INVARIANT VIOLATION at step {step} !!!")
                        log(f"  total distance = {d:.4f} > bound {bound:.4f}")
                        raise RuntimeError("Admissibility invariant broken")

                    last_full_check = step
                    prev_total_tokens = curr_total

                # Light progress print
                if verbose and step % 50 == 0:
                    mstr = " ".join(f"{x:2d}" for x in marking)
                    log(f" {step:4d} | t{t} → [{mstr}]   r ≈ {r:.3f}")

                break

        if not fired:
            log(f"\n→ Terminated naturally after {len(trace)} steps (no enabled transitions)")
            break

    # ── Final summary ───────────────────────────────────────────────────────
    final_r = math.hypot(*embed_marking(marking))
    final_d = poincare_distance(embed_marking(net.initial_marking), embed_marking(marking))
    ok, _, bound = admissible(net.initial_marking, marking, len(trace))

    log("─" * 66)
    log("FINAL RESULT")
    print_marking_bars(marking)
    log("")
    log(f"  steps performed:           {len(trace)}")
    log(f"  final radius:              {final_r:.4f}")
    log(f"  total hyperbolic distance: {final_d:.4f}  ≤  {bound:.4f}   {'OK' if ok else 'VIOLATED'}")
    log(f"  fingerprint (SHA3-256 / 128-bit prefix):")
    log(f"    {fingerprint(net, trace, marking)}")

    print_trajectory_summary(radii, total_distances)

    if output_prefix:
        summary = {
            "timestamp_utc": ts,
            "places": net.n_places,
            "transitions": net.n_trans,
            "initial_marking": net.initial_marking,
            "final_marking": marking,
            "trace": trace,
            "fingerprint": fingerprint(net, trace, marking),
            "radii": [round(r, 4) for r in radii],
            "total_distances": [round(d, 4) for d in total_distances],
            "terminated_naturally": len(trace) < max_steps
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"\nResults saved → {output_prefix}.txt  +  {json_path}")

    if out_file:
        out_file.close()

    optional_plot(step_numbers, radii, total_distances, output_prefix)


# ─── Example: buffered producer–consumer net ────────────────────────────────

EXAMPLE_NET = PetriNet(
    incidence=[
        [-1,  1,  0,  0],   # p0: producer / buffer-in
        [ 1, -1, -1,  0],   # p1: buffer
        [ 0,  1, -1, -1],   # p2: consumer side
        [ 0,  0,  1,  0]    # p3: completed
    ],
    initial_marking=[20, 10, 0, 0]
)


def main():
    parser = argparse.ArgumentParser(description="Hyperbolic Petri Net Simulator")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="output prefix (saves .txt + .json + .png)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="maximum simulation steps (default: 1000)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress console output")
    args = parser.parse_args()

    run_simulation(
        EXAMPLE_NET,
        max_steps=args.max_steps,
        output_prefix=args.out,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
