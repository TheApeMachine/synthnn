"""
Wave/phase-based associative memory for SynthNN.

This implements a complex-valued (phase) associative memory where stored patterns
are unit phasors (complex numbers on the unit circle). Recall is performed by
initializing a resonant network from a cue, letting it settle, and then snapping
to the nearest stored attractor.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .resonant_network import ResonantNetwork
from .resonant_node import ResonantNode


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError("patterns must be 1D or 2D array-like")
    return x


def _to_phasors(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Convert an array to unit phasors.

    - If complex: normalize magnitude to 1 (zeros -> 1+0j)
    - If real: interpret values as radians and take exp(1j*theta)
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        mag = np.abs(x)
        out = np.empty_like(x, dtype=np.complex128)
        nz = mag > eps
        out[nz] = x[nz] / mag[nz]
        out[~nz] = 1.0 + 0.0j
        return out
    # real -> angles
    return np.exp(1j * x.astype(np.float64))


def _project_unit(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Project complex vector to unit circle elementwise (zeros -> 1+0j)."""
    mag = np.abs(x)
    out = np.empty_like(x, dtype=np.complex128)
    nz = mag > eps
    out[nz] = x[nz] / mag[nz]
    out[~nz] = 1.0 + 0.0j
    return out


def _project_unit_or_zero(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Project complex vector to unit circle elementwise, but preserve true unknowns.

    Any element with magnitude <= eps is left as 0+0j instead of being biased to 1+0j.
    """
    mag = np.abs(x)
    out = np.zeros_like(x, dtype=np.complex128)
    nz = mag > eps
    out[nz] = x[nz] / mag[nz]
    return out


def _mean_phase_delta(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Mean absolute phase change between two complex vectors, ignoring magnitude.
    Returns a value in [0, pi].
    """
    # If parts of the state are "unknown" (near-zero magnitude), ignore them so we
    # don't report artificial convergence due to zeros always having angle 0.
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    valid = (np.abs(a) > eps) & (np.abs(b) > eps)
    if not np.any(valid):
        return 0.0
    d = np.angle(a[valid] * np.conj(b[valid]))
    return float(np.mean(np.abs(d)))


@dataclass(frozen=True)
class RecallResult:
    label: str | None
    index: int | None
    score: float
    scores: np.ndarray  # (K,)
    masked_scores: np.ndarray | None  # (K,) or None
    selection: str  # "full" or "rerank(masked->full)"
    final_state: np.ndarray  # (N,), complex
    snapped_state: np.ndarray | None  # (N,), complex
    steps_run: int
    converged: bool
    mean_phase_delta: float


class PhaseAssociativeMemory:
    """
    Complex-phase associative memory implemented on top of ResonantNetwork.

    Patterns are stored as unit phasors (complex values on the unit circle).
    The weight matrix is the sum of outer products of stored patterns.
    """

    def __init__(
        self,
        num_units: int,
        *,
        coupling_strength: float = 0.25,
        damping: float = 0.02,
        zero_diag: bool = True,
        clamp_cue: bool = True,
        project_each_step: bool = True,
        projection_eps: float = 1e-12,
        project_interval: int = 1,
        clamp_alpha: float | None = None,
        node_prefix: str = "mem_",
    ):
        self.num_units: int = int(num_units)
        if self.num_units <= 0:
            raise ValueError("num_units must be > 0")

        self.coupling_strength: float = float(coupling_strength)
        self.damping: float = float(damping)
        self.zero_diag: bool = bool(zero_diag)
        self.clamp_cue: bool = bool(clamp_cue)
        self.project_each_step: bool = bool(project_each_step)
        self.projection_eps: float = float(projection_eps)
        self.project_interval: int = int(project_interval)
        if self.project_interval < 1:
            self.project_interval = 1
        self.clamp_alpha: float | None = None if clamp_alpha is None else float(clamp_alpha)
        if self.clamp_alpha is not None:
            # Soft clamp weight (0..1]. Using 1.0 reproduces a hard override.
            self.clamp_alpha = float(np.clip(self.clamp_alpha, 0.0, 1.0))
        self.node_prefix: str = str(node_prefix)

        self._patterns: np.ndarray | None = None  # (K, N), complex
        self._labels: list[str] | None = None
        self._w: np.ndarray | None = None  # (N, N), complex

        self._network: ResonantNetwork | None = None

    @property
    def patterns(self) -> np.ndarray | None:
        return None if self._patterns is None else self._patterns.copy()

    @property
    def labels(self) -> list[str] | None:
        return None if self._labels is None else list(self._labels)

    def store(self, patterns: np.ndarray, labels: list[str] | None = None) -> None:
        pats = _as_2d(patterns)
        if pats.shape[1] != self.num_units:
            raise ValueError(f"patterns must have shape (K, {self.num_units})")

        pats = _to_phasors(pats)

        K = int(pats.shape[0])
        if labels is None:
            labels = [f"p{i}" for i in range(K)]
        if len(labels) != K:
            raise ValueError("labels length must match number of patterns")

        # Complex Hebbian / holographic rule:
        # W = (1/N) * sum_k p_k p_k^H
        w = (pats.conj().T @ pats) / float(self.num_units)  # (N, N)
        if self.zero_diag:
            np.fill_diagonal(w, 0.0 + 0.0j)

        self._patterns = pats.astype(np.complex128, copy=False)
        self._labels = list(labels)
        self._w = w.astype(np.complex128, copy=False)

        # (Re)build the network substrate.
        self._network = self._build_network(w=self._w)

    def _node_id(self, i: int) -> str:
        return f"{self.node_prefix}{i}"

    def _build_network(self, *, w: np.ndarray) -> ResonantNetwork:
        net = ResonantNetwork(name="phase_associative_memory")
        net.coupling_strength = float(self.coupling_strength)
        net.global_damping = float(self.damping)

        # Nodes: frequency 0 prevents free rotation drifting phase between steps.
        for i in range(self.num_units):
            net.add_node(ResonantNode(self._node_id(i), frequency=0.0, phase=0.0, amplitude=1.0, damping=self.damping))

        # Fully connect using complex weights (works with current Connection.propagate implementation).
        for j in range(self.num_units):
            src = self._node_id(j)
            for i in range(self.num_units):
                if self.zero_diag and i == j:
                    continue
                wij = w[i, j]  # note: coupling sums over inputs (src -> tgt), so weight maps j->i
                if wij == 0:
                    continue
                net.connect(src, self._node_id(i), weight=wij, delay=0.0)

        return net

    def recall(
        self,
        cue: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        steps: int = 200,
        dt: float = 0.02,
        snap: bool = True,
        rerank_top_k: int = 0,
        tol: float = 1e-3,
        patience: int = 8,
    ) -> RecallResult:
        if self._patterns is None or self._labels is None or self._w is None or self._network is None:
            raise RuntimeError("No patterns stored. Call store() first.")

        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be > 0")

        cue_v = np.asarray(cue)
        if cue_v.ndim != 1 or cue_v.shape[0] != self.num_units:
            raise ValueError(f"cue must have shape ({self.num_units},)")

        cue_ph = _to_phasors(cue_v)

        if mask is None:
            known = np.ones(self.num_units, dtype=bool)
        else:
            known = np.asarray(mask, dtype=bool)
            if known.shape != (self.num_units,):
                raise ValueError(f"mask must have shape ({self.num_units},)")

        def _apply_clamp() -> None:
            """Apply cue constraints to known units."""
            if not self.clamp_cue:
                return
            alpha = self.clamp_alpha
            for i in range(self.num_units):
                if not known[i]:
                    continue
                nid = self._node_id(i)
                if alpha is None or alpha >= 1.0:
                    # Hard clamp: hold exactly at the cue (Dirichlet-like boundary).
                    self._network.nodes[nid].signal = complex(cue_ph[i])
                else:
                    # Soft clamp: blend current state toward cue.
                    s = self._network.nodes[nid].signal
                    self._network.nodes[nid].signal = (1.0 - alpha) * s + alpha * complex(cue_ph[i])

        # Initialize node signals from cue; unknown units start at 0 (no direction).
        for i in range(self.num_units):
            nid = self._node_id(i)
            if known[i]:
                self._network.nodes[nid].signal = complex(cue_ph[i])
            else:
                self._network.nodes[nid].signal = 0.0 + 0.0j

        # Ensure clamped units start exactly on the cue (important if clamp_alpha is used).
        _apply_clamp()

        prev_state = _project_unit_or_zero(
            np.array([self._network.nodes[self._node_id(i)].signal for i in range(self.num_units)], dtype=np.complex128),
            eps=self.projection_eps,
        )

        stable = 0
        converged = False
        last_delta = float("inf")

        for t in range(steps):
            # Apply constraints before stepping (so couplings see the constrained boundary).
            _apply_clamp()

            self._network.step(float(dt))

            if self.project_each_step:
                if (t % self.project_interval) == 0:
                    for i in range(self.num_units):
                        nid = self._node_id(i)
                        self._network.nodes[nid].signal = complex(
                            _project_unit_or_zero(
                                np.array([self._network.nodes[nid].signal], dtype=np.complex128),
                                eps=self.projection_eps,
                            )[0]
                        )

            # Re-apply clamp after stepping/projection so known units are truly held fixed
            # during convergence checking and scoring.
            _apply_clamp()

            cur_state = _project_unit_or_zero(
                np.array([self._network.nodes[self._node_id(i)].signal for i in range(self.num_units)], dtype=np.complex128),
                eps=self.projection_eps,
            )

            last_delta = _mean_phase_delta(cur_state, prev_state, eps=self.projection_eps)
            if last_delta < float(tol):
                stable += 1
                if stable >= int(patience):
                    converged = True
                    prev_state = cur_state
                    steps_run = t + 1
                    break
            else:
                stable = 0

            prev_state = cur_state
        else:
            steps_run = steps

        final_state = prev_state

        # Score against stored patterns. Use magnitude of inner product to ignore global phase rotation.
        pats = self._patterns  # (K, N)
        scores = np.abs(pats.conj() @ final_state) / float(self.num_units)  # (K,)

        selection = "full"
        masked_scores: np.ndarray | None = None

        # Optional rerank for partial cues:
        # 1) score on known dimensions only (invariant to global rotation)
        # 2) take top-k by masked score
        # 3) choose the best full-score among those candidates (uses recovered dims as tie-break)
        if rerank_top_k and (mask is not None) and np.any(~known) and scores.size:
            m = int(np.sum(known))
            if m > 0:
                cue_known = cue_ph[known]
                masked_scores = np.abs(pats[:, known].conj() @ cue_known) / float(m)  # (K,)
                k = int(min(max(int(rerank_top_k), 1), masked_scores.size))
                # Indices of top-k masked scores (unordered).
                cand = np.argpartition(masked_scores, -k)[-k:]
                # Prefer best masked score; use full-score only as a tie-break.
                best_m = float(np.max(masked_scores[cand]))
                tie = cand[masked_scores[cand] >= (best_m - 1e-12)]
                best_idx = int(tie[np.argmax(scores[tie])])
                selection = "rerank(masked->full)"
            else:
                best_idx = int(np.argmax(scores)) if scores.size else None
        else:
            best_idx = int(np.argmax(scores)) if scores.size else None

        best_full_score = float(scores[best_idx]) if best_idx is not None else 0.0
        if best_idx is not None and selection.startswith("rerank") and masked_scores is not None:
            best_score = float(masked_scores[best_idx])
        else:
            best_score = best_full_score
        best_label = self._labels[best_idx] if best_idx is not None else None

        snapped_state: np.ndarray | None = None
        if snap and best_idx is not None:
            # Align global phase so snapped pattern is closest to final_state.
            ph = np.vdot(pats[best_idx], final_state)  # conj(p)·final
            rot = np.exp(1j * np.angle(ph)) if ph != 0 else (1.0 + 0.0j)
            snapped_state = (pats[best_idx] * rot).astype(np.complex128, copy=False)

        return RecallResult(
            label=best_label,
            index=best_idx,
            score=best_score,
            scores=scores.astype(np.float64, copy=False),
            masked_scores=None if masked_scores is None else masked_scores.astype(np.float64, copy=False),
            selection=selection,
            final_state=final_state.astype(np.complex128, copy=False),
            snapped_state=snapped_state,
            steps_run=int(steps_run),
            converged=bool(converged),
            mean_phase_delta=float(last_delta if np.isfinite(last_delta) else 0.0),
        )

    def save(self, path: str) -> None:
        if self._patterns is None or self._labels is None or self._w is None:
            raise RuntimeError("No patterns stored. Call store() first.")
        np.savez_compressed(
            path,
            num_units=np.array([self.num_units], dtype=np.int64),
            patterns=self._patterns.astype(np.complex128),
            labels=np.array(self._labels, dtype=object),
            W=self._w.astype(np.complex128),
            coupling_strength=np.array([self.coupling_strength], dtype=np.float64),
            damping=np.array([self.damping], dtype=np.float64),
            zero_diag=np.array([int(self.zero_diag)], dtype=np.int8),
            clamp_cue=np.array([int(self.clamp_cue)], dtype=np.int8),
            project_each_step=np.array([int(self.project_each_step)], dtype=np.int8),
            projection_eps=np.array([self.projection_eps], dtype=np.float64),
            project_interval=np.array([self.project_interval], dtype=np.int64),
            clamp_alpha=np.array([(-1.0 if self.clamp_alpha is None else float(self.clamp_alpha))], dtype=np.float64),
            node_prefix=np.array([self.node_prefix], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "PhaseAssociativeMemory":
        d = np.load(path, allow_pickle=True)
        num_units = int(np.asarray(d["num_units"]).reshape(-1)[0])
        # Optional fields added later; fall back safely for old files.
        projection_eps = float(np.asarray(d["projection_eps"]).reshape(-1)[0]) if "projection_eps" in d else 1e-12
        project_interval = int(np.asarray(d["project_interval"]).reshape(-1)[0]) if "project_interval" in d else 1
        clamp_alpha_raw = float(np.asarray(d["clamp_alpha"]).reshape(-1)[0]) if "clamp_alpha" in d else -1.0
        clamp_alpha = None if clamp_alpha_raw < 0 else float(clamp_alpha_raw)

        mem = cls(
            num_units=num_units,
            coupling_strength=float(np.asarray(d["coupling_strength"]).reshape(-1)[0]),
            damping=float(np.asarray(d["damping"]).reshape(-1)[0]),
            zero_diag=bool(int(np.asarray(d["zero_diag"]).reshape(-1)[0])),
            clamp_cue=bool(int(np.asarray(d["clamp_cue"]).reshape(-1)[0])),
            project_each_step=bool(int(np.asarray(d["project_each_step"]).reshape(-1)[0])),
            projection_eps=projection_eps,
            project_interval=project_interval,
            clamp_alpha=clamp_alpha,
            node_prefix=str(np.asarray(d["node_prefix"]).reshape(-1)[0]),
        )
        patterns = np.asarray(d["patterns"])
        labels = [str(x) for x in np.asarray(d["labels"]).tolist()]
        mem.store(patterns, labels=labels)
        return mem

