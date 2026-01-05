import unittest
import numpy as np
import tempfile
import os


class TestPhaseAssociativeMemory(unittest.TestCase):
    def setUp(self):
        # Local import to keep test startup cheap
        from synthnn.core.associative_memory import PhaseAssociativeMemory

        self.PhaseAssociativeMemory = PhaseAssociativeMemory
        self.rng = np.random.default_rng(42)

    def _random_patterns(self, k: int, n: int) -> np.ndarray:
        angles = self.rng.uniform(-np.pi, np.pi, size=(k, n)).astype(np.float64)
        return np.exp(1j * angles)

    def test_recall_from_phase_noise(self):
        N = 48
        K = 4
        patterns = self._random_patterns(K, N)
        labels = [f"pat{i}" for i in range(K)]

        mem = self.PhaseAssociativeMemory(
            N,
            coupling_strength=0.35,
            damping=0.02,
            zero_diag=True,
            clamp_cue=True,
            project_each_step=True,
        )
        mem.store(patterns, labels=labels)

        target = 2
        base = patterns[target]
        noise = self.rng.normal(0.0, 0.35, size=N)  # radians
        cue = np.exp(1j * (np.angle(base) + noise))

        res = mem.recall(cue, steps=300, dt=0.05, snap=True)
        self.assertEqual(res.label, labels[target])
        self.assertGreater(res.score, 0.35)

    def test_recall_from_partial_cue(self):
        N = 64
        K = 3
        patterns = self._random_patterns(K, N)
        labels = [f"pat{i}" for i in range(K)]

        mem = self.PhaseAssociativeMemory(
            N,
            coupling_strength=0.40,
            damping=0.02,
            zero_diag=True,
            clamp_cue=True,
            project_each_step=True,
        )
        mem.store(patterns, labels=labels)

        target = 1
        base = patterns[target]

        # Only keep 40% of units as known
        mask = self.rng.random(N) < 0.40
        cue = base.copy()
        cue[~mask] = 1.0 + 0.0j  # ignored because mask tells recall what is known

        res = mem.recall(cue, mask=mask, steps=400, dt=0.05, snap=True)
        self.assertEqual(res.label, labels[target])
        self.assertGreater(res.score, 0.30)

    def test_persistence_round_trip(self):
        N = 32
        K = 3
        patterns = self._random_patterns(K, N)
        labels = [f"pat{i}" for i in range(K)]

        mem = self.PhaseAssociativeMemory(N, coupling_strength=0.3, damping=0.02)
        mem.store(patterns, labels=labels)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mem.npz")
            mem.save(path)

            mem2 = self.PhaseAssociativeMemory.load(path)

            self.assertEqual(mem2.num_units, mem.num_units)
            self.assertEqual(mem2.labels, mem.labels)

            # Verify recall still works after loading
            res = mem2.recall(patterns[0], steps=200, dt=0.05, snap=True)
            self.assertEqual(res.label, labels[0])


if __name__ == "__main__":
    unittest.main()

