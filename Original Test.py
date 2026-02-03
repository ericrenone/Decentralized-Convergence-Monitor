#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperbolic Petri Embedding (HPE) — Cleaned & Error-Free Single-File Version

Features:
- Poincaré disk embedding heuristic for markings
- Hyperbolic distance tracking + weak admissibility invariant
- Rich console visualization (ASCII bars, trajectory plot)
- Fixed deprecation warning (uses timezone.utc)
- JSON + fingerprint for reproducibility
- Greedy first-enabled transition firing

Author: Adapted from Eric Ren's version
License: MIT
Date: February 2026
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# I. Petri Net Definition
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PetriNet:
    incidence: List[List[int]]
    initial_marking: List[int]

    @property
    def n_places(self) -> int:
        return len(self.initial_marking)

    @property
    def n_trans(self) -> int:
        return len(self.incidence[0]) if self.incidence else 0

    def can_fire(self, t: int, m: List[int]) -> bool:
        return all(m[p] + self.incidence[p][t] >= 0 for p in range(self.n_places))

    def fire(self, t: int, m: List[int]) -> List[int]:
        if not self.can_fire(t, m):
            raise ValueError(f"Transition t{t} not enabled")
        return [m[p] + self.incidence[p][t] for p in range(self.n_places)]


# ─────────────────────────────────────────────────────────────────────
# II. Poincaré Disk Embedding
# ─────────────────────────────────────────────────────────────────────

def embed_marking(m: List[int], radius_scale: float = 0.32) -> Tuple[float, float]:
    total = sum(m)
    if total == 0:
        return 0.0, 0.0

    theta = 0.0
    weight_sum = 0.0
    n = len(m)
    for i, cnt in enumerate(m):
        if cnt == 0:
            continue
        frac = cnt / total
        angle = 2.0 * math.pi * i / max(1, n - 1)
        theta += frac * angle
        weight_sum += frac

    if weight_sum > 0:
        theta /= weight_sum

    r = math.tanh(radius_scale * math.log1p(total))
    return r * math.cos(theta), r * math.sin(theta)


def poincare_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    aa = ax*ax + ay*ay
    bb = bx*bx + by*by
    diff2 = (ax - bx)**2 + (ay - by)**2
    denom = (1 - aa) * (1 - bb)
    if denom < 1e-12:
        return 999.0
    arg = 1.0 + 2.0 * diff2 / denom
    return math.acosh(max(arg, 1.0))


# ─────────────────────────────────────────────────────────────────────
# III. Heuristics & Invariant
# ─────────────────────────────────────────────────────────────────────

def radial_ratio(prev: Tuple[float, float], curr: Tuple[float, float]) -> float:
    rp = math.hypot(*prev) + 1e-14
    rc = math.hypot(*curr)
    return rc / rp


def admissible(m0: List[int], mk: List[int], steps: int, factor: float = 0.9) -> Tuple[bool, float, float]:
    d = poincare_distance(embed_marking(m0), embed_marking(mk))
    bound = factor * 2.8 * math.log1p(steps + 1)
    return d <= bound, d, bound


# ─────────────────────────────────────────────────────────────────────
# IV. Visualization Helpers
# ─────────────────────────────────────────────────────────────────────

def ascii_bar(value: float, max_val: float, width: int = 20, char: str = "█") -> str:
    if max_val <= 0:
        return " " * width
    filled = int(width * min(value, max_val) / max_val)
    return char * filled + " " * (width - filled)


def print_trajectory(radii: List[float], distances: List[float], width: int = 60):
    if not radii:
        return
    max_r = max(radii) + 1e-6
    max_d = max(distances) + 1e-6

    print("  Hyperbolic trajectory (radius ─── total distance from origin)")
    for i, (r, d) in enumerate(zip(radii, distances)):
        r_bar = ascii_bar(r, max_r, width // 2 - 2)
        d_bar = ascii_bar(d, max_d, width // 2 - 2)
        print(f" {i+1:2d} | {r_bar} {r:5.3f} | {d_bar} {d:5.3f}")


def print_marking_bars(m: List[int], labels: Optional[List[str]] = None):
    if not m:
        return
    mx = max(m) + 1
    lbl = labels or [f"p{i}" for i in range(len(m))]
    print("  Current marking distribution:")
    for i, (cnt, name) in enumerate(zip(m, lbl)):
        bar = ascii_bar(cnt, mx, 24)
        print(f"    {name:>4} | {bar} {cnt}")


def optional_matplotlib_plot(
    steps: List[int],
    radii: List[float],
    total_d: List[float],
    filename: Optional[str] = None
):
    if not MATPLOTLIB_AVAILABLE or not steps:
        print("  (matplotlib not available or no data — skipping plot)")
        return

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    ax1.set_xlabel("Simulation step")
    ax1.set_ylabel("Radial coordinate", color="#1f77b4")
    ax1.plot(steps, radii, "o-", color="#1f77b4", label="radius")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Total hyperbolic distance", color="#ff7f0e")
    ax2.plot(steps, total_d, "s--", color="#ff7f0e", label="total d")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    fig.suptitle("Hyperbolic Petri Embedding Trajectory", fontsize=13)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if filename:
        plt.savefig(filename + "_trajectory.png", dpi=150, bbox_inches="tight")
        print(f"  Saved plot → {filename}_trajectory.png")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────
# V. Fingerprint
# ─────────────────────────────────────────────────────────────────────

def fingerprint(net: PetriNet, trace: List[int], final: List[int]) -> str:
    h = hashlib.sha3_512()
    for row in net.incidence:
        for v in row:
            h.update(v.to_bytes(2, "little", signed=True))
    for t in trace:
        h.update(t.to_bytes(2, "little", signed=False))
    for m in final:
        h.update(m.to_bytes(4, "little", signed=False))
    return h.hexdigest()[:64]


# ─────────────────────────────────────────────────────────────────────
# VI. Main Simulation
# ─────────────────────────────────────────────────────────────────────

def run_simulation(
    net: PetriNet,
    max_steps: int = 100,
    output_prefix: Optional[str] = None,
    verbose: bool = True
):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_file = None
    json_path = None

    if output_prefix:
        out_file = open(output_prefix + ".txt", "w", encoding="utf-8")
        json_path = output_prefix + ".json"

    def log(msg: str):
        if verbose:
            print(msg)
        if out_file:
            out_file.write(msg + "\n")
            out_file.flush()

    log("╔══════════════════════════════════════════════════════╗")
    log("║        Hyperbolic Petri Embedding  (2026)            ║")
    log("╚══════════════════════════════════════════════════════╝")
    log(f"UTC: {ts}")
    log(f"Places: {net.n_places}   Transitions: {net.n_trans}")
    log(f"Initial: {net.initial_marking}")
    log("─" * 55)

    marking = net.initial_marking[:]
    trace: List[int] = []
    radii: List[float] = []
    total_ds: List[float] = []
    step_list: List[int] = []

    terminated = False
    for step in range(1, max_steps + 1):
        fired = False
        for t in range(net.n_trans):
            if net.can_fire(t, marking):
                prev_embed = embed_marking(marking)
                prev_r = math.hypot(*prev_embed)

                marking = net.fire(t, marking)
                trace.append(t)
                fired = True

                curr_embed = embed_marking(marking)
                curr_r = math.hypot(*curr_embed)
                delta_d = poincare_distance(prev_embed, curr_embed)
                ratio = radial_ratio(prev_embed, curr_embed)
                ok, total_d, bound = admissible(net.initial_marking, marking, len(trace))

                # Clean marking display
                marking_str = "[" + ", ".join(f"{x:2}" for x in marking) + "]"
                log(f"Step {step:3d} | t{t} → {marking_str}  Δd={delta_d:5.3f}  r={curr_r:5.3f}")
                log(f"         radial ratio={ratio:5.3f}   total d={total_d:5.3f} ≤ {bound:5.3f}  {'✓' if ok else '✗'}")

                step_list.append(len(trace))
                radii.append(curr_r)
                total_ds.append(total_d)

                if not ok:
                    log("\n!!! HYPERBOLIC DISPLACEMENT VIOLATION !!!\n")
                    raise RuntimeError("Admissibility invariant broken")

                break

        if not fired:
            log("\n→ TERMINATED NATURALLY (no enabled transitions)")
            terminated = True
            break

        if step % 8 == 0 or step == max_steps or terminated:
            log("\n" + "═" * 55)
            print_marking_bars(marking)
            log("")
            print_trajectory(radii, total_ds)
            log("═" * 55 + "\n")

    fp = fingerprint(net, trace, marking)
    log("Final marking: " + str(marking))
    log(f"Trace length: {len(trace)}   Trace: {trace}")
    log(f"Structural fingerprint (SHA3-512 prefix):\n{fp}")

    if output_prefix:
        summary = {
            "timestamp_utc": ts,
            "net": {"places": net.n_places, "trans": net.n_trans, "incidence": net.incidence},
            "initial": net.initial_marking,
            "final": marking,
            "trace": trace,
            "fingerprint": fp,
            "terminated": terminated,
            "radii": radii,
            "total_distances": total_ds
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"\nResults saved → {output_prefix}.txt  +  {json_path}")

    if out_file:
        out_file.close()

    # Final summary visualization
    print("\n" + "═" * 55)
    print("FINAL VISUAL SUMMARY")
    print_marking_bars(marking)
    print("")
    print_trajectory(radii, total_ds)
    optional_matplotlib_plot(step_list, radii, total_ds, output_prefix)


# ─────────────────────────────────────────────────────────────────────
# Example Net (buffered producer–consumer)
# ─────────────────────────────────────────────────────────────────────

EXAMPLE_NET = PetriNet(
    incidence=[
        [-1,  1,  0,  0],   # p0: producer / buffer-in
        [ 1, -1, -1,  0],   # p1: buffer
        [ 0,  1, -1, -1],   # p2: consumer side
        [ 0,  0,  1,  0]    # p3: completed
    ],
    initial_marking=[5, 2, 0, 0]
)


def main():
    parser = argparse.ArgumentParser(description="Hyperbolic Petri Net Explorer")
    parser.add_argument("--out", "-o", type=str, default=None, help="output prefix")
    parser.add_argument("--max-steps", type=int, default=120, help="maximum simulation steps")
    parser.add_argument("--quiet", action="store_true", help="suppress console output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("\nRunning built-in example Petri net...\n")

    run_simulation(
        EXAMPLE_NET,
        max_steps=args.max_steps,
        output_prefix=args.out,
        verbose=verbose
    )


if __name__ == "__main__":
    main()
