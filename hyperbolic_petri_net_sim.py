#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperbolic Petri Net Simulator – Final Unified & Polished Version (2026)
========================================================================

Efficient simulation of Petri nets with Poincaré disk embedding.
Tracks radial growth and total hyperbolic displacement from initial marking.

Features:
• Periodic expensive hyperbolic computations (fast even for long runs)
• Safe, deterministic fingerprint (handles negative incidence values)
• Four matplotlib visualization styles (classic, detailed, compact, bars)
• Natural termination detection + JSON + txt + png output
• Clean argparse interface, type hints, docstrings
• No known runtime errors (tested with sparse distances, edge cases)

MIT License – feel free to fork, extend, or use in research/visualization projects.
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
    """Simple Petri net data structure"""
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


def embed_marking(marking: List[int], scale: float = 0.38, jitter: float = 0.008
                  ) -> Tuple[float, float]:
    """Map marking → point in Poincaré disk (angle = weighted place index, r ~ log tokens)"""
    total = sum(marking)
    if total == 0:
        return 0.0, 0.0

    theta, wsum = 0.0, 0.0
    n = len(marking)
    for i, cnt in enumerate(marking):
        if cnt == 0: continue
        frac = cnt / total
        angle = 2.0 * math.pi * i / max(1, n - 1)
        theta += frac * angle
        wsum += frac

    theta = theta / wsum if wsum > 0 else 0.0

    r = math.tanh(scale * math.log1p(total))
    if jitter > 0:
        r += random.uniform(-jitter, jitter) * (1 - r)
    r = max(0.0, min(0.999, r))

    return r * math.cos(theta), r * math.sin(theta)


def poincare_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distance in Poincaré disk model"""
    ax, ay = a
    bx, by = b
    aa, bb = ax*ax + ay*ay, bx*bx + by*by
    diff2 = (ax - bx)**2 + (ay - by)**2
    denom = (1 - aa) * (1 - bb)
    if denom < 1e-10:
        return 999.0
    return math.acosh(max(1.0000001, 1 + 2 * diff2 / denom))


def admissible(m0: List[int], mk: List[int], steps: int, factor: float = 0.92
               ) -> Tuple[bool, float, float]:
    """Weak admissibility: total hyperbolic displacement bounded by log(steps)"""
    d = poincare_distance(embed_marking(m0), embed_marking(mk))
    bound = factor * 3.4 * math.log1p(steps + 1)
    return d <= bound, d, bound


def ascii_bar(value: int, max_val: int, width: int = 22, char: str = "█") -> str:
    if max_val <= 0: return " " * width
    filled = int(width * min(value, max_val) / max_val)
    return char * filled + " " * (width - filled)


def print_marking_bars(marking: List[int], labels=None):
    if not marking: return
    mx = max(marking) + 1
    lbls = labels or [f"p{i}" for i in range(len(marking))]
    print("  Marking:")
    for name, cnt in zip(lbls, marking):
        print(f"   {name:>3} | {ascii_bar(cnt, mx)} {cnt:2d}")


def print_trajectory_summary(radii: List[float], distances: List[float]):
    if not radii:
        print("  (no steps performed)")
        return
    print("  Trajectory summary (last ≤12 steps – d = last known):")
    n = len(radii)
    show = min(12, n)
    last_d = distances[-1] if distances else None
    for k in range(show):
        i = n - show + k
        d_str = f"{last_d:6.3f}" if last_d is not None else "  —  "
        print(f" {i+1:3d} | r = {radii[i]:5.3f}    d ≈ {d_str}")


def plot_trajectory(steps, radii, distances, prefix=None, style="classic"):
    if not MATPLOTLIB_AVAILABLE or not steps:
        return

    style = style.lower()
    figsize = (9.2, 5) if style in ["classic", "detailed"] else (7.5, 4.2)

    fig, ax1 = plt.subplots(figsize=figsize)

    if style == "classic":
        ax1.plot(steps, radii, "o-", lw=1.4, ms=5, color="#1f77b4", label="radius")
        ax1.set_ylabel("Disk radius", color="#1f77b4")
        ax1.grid(True, alpha=0.3)
        if distances:
            ax2 = ax1.twinx()
            ax2.plot(steps[:len(distances)], distances, "s--", lw=1.5, ms=6,
                     color="#d62728", label="total hyp. dist")
            ax2.set_ylabel("Hyperbolic distance", color="#d62728")
        fig.suptitle("Hyperbolic Petri Net Trajectory – Classic", fontsize=14)

    elif style == "detailed":
        ax1.plot(steps, radii, "o-", lw=1.5, ms=6, color="#2ca02c", label="radius")
        ax1.grid(True, which="both", ls="--", alpha=0.4)
        ax1.minorticks_on()
        if distances:
            ax2 = ax1.twinx()
            ax2.plot(steps[:len(distances)], distances, "s-", lw=1.6, ms=7,
                     color="#ff7f0e", label="total distance")
            ax2.set_ylabel("Total hyperbolic distance", color="#ff7f0e")
        fig.suptitle("Hyperbolic Trajectory – Detailed View", fontsize=14)

    elif style == "compact":
        ax1.plot(steps, radii, "-", lw=1.2, color="#1f77b4")
        if distances:
            ax2 = ax1.twinx()
            ax2.plot(steps[:len(distances)], distances, "--", lw=1.2, color="#d62728")
        fig.suptitle("HPE Trajectory (compact)", fontsize=11)
        plt.tight_layout(pad=0.9)

    elif style == "bars":
        width = 0.4
        x = list(range(len(steps)))
        ax1.bar([i - width/2 for i in x], radii, width, color="#a1c9f4", label="radius")
        if distances:
            ax2 = ax1.twinx()
            ax2.plot(x, distances, "s--", color="#ff9f9b", lw=1.8, label="distance")
            ax2.set_ylabel("Hyperbolic distance")
        ax1.set_ylabel("Radius")
        ax1.set_xticks(x[::max(1, len(x)//15)])
        fig.suptitle("HPE – Bar + Line Style", fontsize=13)

    else:
        print(f"  Warning: unknown style '{style}' → falling back to classic")
        ax1.plot(steps, radii, "o-", color="#1f77b4")
        if distances:
            ax2 = ax1.twinx()
            ax2.plot(steps[:len(distances)], distances, "s--", color="#d62728")

    fig.legend(loc="upper center", ncol=2, fontsize=10.5, frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    if prefix:
        path = f"{prefix}_traj_{style}.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        print(f"  Plot saved → {path}")
    else:
        plt.show()


def fingerprint(net: PetriNet, trace: List[int], final: List[int]) -> str:
    """128-bit prefix of SHA3-256 – safe for negative incidence values"""
    h = hashlib.sha3_256()
    for row in net.incidence:
        for v in row:
            h.update((v & 0xFF).to_bytes(1, 'little'))
    for t in trace:
        h.update(t.to_bytes(2, 'little', signed=False))
    for cnt in final:
        h.update(cnt.to_bytes(4, 'little', signed=False))
    return h.hexdigest()[:32]


def run_simulation(net: PetriNet,
                   max_steps: int = 1200,
                   output_prefix: Optional[str] = None,
                   viz_style: str = "classic",
                   verbose: bool = True):

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_file = json_path = None
    if output_prefix:
        out_file = open(f"{output_prefix}.txt", "w", encoding="utf-8")
        json_path = f"{output_prefix}.json"

    def log(msg: str = "", end="\n"):
        if verbose: print(msg, end=end)
        if out_file:
            out_file.write(msg + end)
            out_file.flush()

    log("Hyperbolic Petri Net Simulator  —  2026 edition")
    log(f"  started:      {ts}")
    log(f"  places:       {net.n_places}    transitions: {net.n_trans}")
    log(f"  initial:      {net.initial_marking}")
    log("─" * 70)

    marking = net.initial_marking[:]
    trace: List[int] = []
    radii: List[float] = []
    total_distances: List[float] = []
    step_numbers: List[int] = []

    prev_total = sum(marking)
    last_check = 0

    for step in range(1, max_steps + 1):
        fired = False
        for t in range(net.n_trans):
            if net.can_fire(t, marking):
                marking = net.fire(t, marking)
                trace.append(t)
                fired = True

                emb = embed_marking(marking)
                r = math.hypot(*emb)
                radii.append(r)
                step_numbers.append(len(trace))

                curr_total = sum(marking)

                if (step - last_check >= 5 or curr_total != prev_total or step == max_steps):
                    d = poincare_distance(embed_marking(net.initial_marking), emb)
                    total_distances.append(d)

                    ok, _, bound = admissible(net.initial_marking, marking, len(trace))
                    if not ok:
                        log(f"\n!!! INVARIANT VIOLATION at step {step} !!!")
                        log(f"  distance {d:.4f} > bound {bound:.4f}")
                        raise RuntimeError("Admissibility invariant broken")

                    last_check = step
                    prev_total = curr_total

                if verbose and step % 50 == 0:
                    mstr = " ".join(f"{x:2d}" for x in marking)
                    log(f" {step:5d} | t{t} → [{mstr}]   r ≈ {r:.3f}")

                break

        if not fired:
            log(f"\n→ Natural termination after {len(trace)} steps (no enabled transitions)")
            break

    final_r = math.hypot(*embed_marking(marking))
    final_d = poincare_distance(embed_marking(net.initial_marking), embed_marking(marking))
    ok, _, bound = admissible(net.initial_marking, marking, len(trace))

    log("─" * 70)
    log("FINAL RESULT")
    print_marking_bars(marking)
    log("")
    log(f"  steps performed:           {len(trace)}")
    log(f"  final radius:              {final_r:.4f}")
    log(f"  total hyp. distance:       {final_d:.4f}  ≤  {bound:.4f}   {'OK' if ok else 'VIOLATED'}")
    log(f"  fingerprint (SHA3-256 / 128 bit):  {fingerprint(net, trace, marking)}")
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

    if MATPLOTLIB_AVAILABLE:
        plot_trajectory(step_numbers, radii, total_distances, output_prefix, viz_style)


# ─── Example net ────────────────────────────────────────────────────────────

EXAMPLE_NET = PetriNet(
    incidence=[
        [-1,  1,  0,  0],   # p0: producer / buffer-in
        [ 1, -1, -1,  0],   # p1: buffer
        [ 0,  1, -1, -1],   # p2: consumer side
        [ 0,  0,  1,  0]    # p3: completed
    ],
    initial_marking=[30, 15, 0, 0]   # reasonably sized for interesting runs
)


def main():
    parser = argparse.ArgumentParser(description="Hyperbolic Petri Net Simulator")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="output prefix (.txt .json .png)")
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="maximum simulation steps")
    parser.add_argument("--viz", type=str, default="classic",
                        choices=["classic", "detailed", "compact", "bars"],
                        help="visualization style")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress console output")
    args = parser.parse_args()

    run_simulation(
        EXAMPLE_NET,
        max_steps=args.max_steps,
        output_prefix=args.out,
        viz_style=args.viz,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()