#!/usr/bin/env python3
"""
Associative memory demo (phase patterns).

This demonstrates storing several complex phase patterns as attractors and
recalling them from:
- a noisy cue (phase jitter)
- a partial cue (mask)
"""

from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
import numpy as np

# Allow running from repo root without installation
sys.path.insert(0, ".")

from synthnn.core import PhaseAssociativeMemory


def _random_patterns(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    angles = rng.uniform(-np.pi, np.pi, size=(k, n)).astype(np.float64)
    return np.exp(1j * angles)


@dataclass
class Args:
    seed: int = 123
    units: int = 64
    dtype: str = "c128"  # "c64" | "c128"
    patterns: int = 5
    targets: int | None = None
    target: int = -1
    noise_std: float = 0.65
    known_frac: float = 0.28
    rerank_top: int = 64
    steps: int = 500
    dt: float = 0.05
    no_labels: bool = False


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase associative memory demo (random target + controllable difficulty).")
    _ = ap.add_argument("--seed", type=int, default=123, help="RNG seed for patterns and corruption.")
    _ = ap.add_argument("--units", type=int, default=64, help="Number of memory units (N).")
    _ = ap.add_argument(
        "--dtype",
        type=str,
        default="c128",
        choices=["c64", "c128"],
        help="Internal complex dtype. c64 is faster/less memory, but exact recall scoring is still O(K*N) and patterns must fit in RAM.",
    )
    _ = ap.add_argument("--patterns", type=int, default=5, help="Number of stored patterns (K). (Alias: --targets)")
    _ = ap.add_argument("--targets", type=int, default=None, help="Alias for --patterns (number of possible targets).")
    _ = ap.add_argument("--target", type=int, default=-1, help="Target pattern index (0..K-1). -1 picks randomly.")
    _ = ap.add_argument("--noise-std", type=float, default=0.65, help="Phase noise std dev in radians for noisy cue.")
    _ = ap.add_argument("--known-frac", type=float, default=0.28, help="Fraction of units revealed for partial cue.")
    _ = ap.add_argument("--rerank-top", type=int, default=64, help="Top-k candidates for partial-cue rerank (0 disables). Uses settled-state masking, not raw cue.")
    _ = ap.add_argument("--steps", type=int, default=500, help="Max settling steps.")
    _ = ap.add_argument("--dt", type=float, default=0.05, help="Time step for settling.")
    _ = ap.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not materialize K Python strings for labels (important for large K).",
    )
    args: Args = ap.parse_args(namespace=Args())

    rng = np.random.default_rng(args.seed)

    N = args.units
    K = args.targets if args.targets is not None else args.patterns
    patterns: np.ndarray = _random_patterns(rng, K, N)
    labels = None if args.no_labels else [f"pattern_{i}" for i in range(K)]

    dtype = np.dtype(np.complex64) if args.dtype == "c64" else np.dtype(np.complex128)
    mem = PhaseAssociativeMemory(
        N,
        dtype=dtype,
        label_prefix="pattern_" if labels is None else None,  # lazy labels if not provided
        coupling_strength=0.35,
        damping=0.02,
        zero_diag=True,
        clamp_cue=True,
        project_each_step=True,
    )
    mem.store(patterns, labels=labels)

    if K >= 100_000:
        print("NOTE: Large K detected. This demo stores all patterns in RAM and recall scoring is O(K*N).")
        print("      For million-scale K, you'll likely need memmapped storage and/or approximate candidate selection.")
        print()

    if args.target < 0:
        target = int(rng.integers(0, int(K)))
    else:
        target = int(args.target)
        if not (0 <= target < K):
            raise SystemExit(f"--target must be in [0, {K-1}] (or -1 for random)")

    base: np.ndarray = patterns[target]

    if labels is not None:
        if K <= 20:
            print("Stored labels:", labels)
        else:
            head = labels[:10]
            tail = labels[-3:]
            print(f"Stored labels: K={K} (showing first 10 + last 3)")
            print(" ", head, "...", tail)
    print("Target label:", mem.label_of(target))
    print()

    # --- Noisy cue ---
    noise_std = float(args.noise_std)  # radians
    cue_noisy: np.ndarray = np.exp(1j * (np.angle(base) + rng.normal(0.0, noise_std, size=N)))
    res1 = mem.recall(cue_noisy, steps=int(args.steps), dt=float(args.dt), snap=True)
    print("Noisy cue")
    print("  noise_std(rad):", noise_std)
    print("  recalled:", res1.label, f"(score={res1.score:.3f}, steps={res1.steps_run}, converged={res1.converged})")
    print()

    # --- Partial cue ---
    known_frac = float(args.known_frac)
    mask = rng.random(N) < known_frac
    cue_partial: np.ndarray = base.copy()
    cue_partial[~mask] = 1.0 + 0.0j
    res2 = mem.recall(cue_partial, mask=mask, steps=int(args.steps), dt=float(args.dt), snap=True)
    print("Partial cue")
    print("  known_frac:", float(np.mean(mask)))
    print("  recalled:", res2.label, f"(score={res2.score:.3f}, steps={res2.steps_run}, converged={res2.converged})")

    # Rerank using only known units, then break ties with full score.
    if int(args.rerank_top) > 0:
        res2r = mem.recall(
            cue_partial,
            mask=mask,
            steps=int(args.steps),
            dt=float(args.dt),
            snap=True,
            rerank_top_k=int(args.rerank_top),
        )
        full_score = float(res2r.scores[res2r.index]) if res2r.index is not None else 0.0
        masked_score = float(res2r.masked_scores[res2r.index]) if (res2r.index is not None and res2r.masked_scores is not None) else 0.0
        print(
            "  reranked:",
            res2r.label,
            f"(masked={masked_score:.3f}, full={full_score:.3f}, selection={res2r.selection})",
        )


if __name__ == "__main__":
    main()

