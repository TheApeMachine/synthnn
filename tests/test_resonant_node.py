import unittest
import numpy as np
import pickle
import sys

# Add root project directory to path
sys.path.insert(0, '.')

from synthnn.core.resonant_node import ResonantNode

class TestResonantNode(unittest.TestCase):
    """Unit tests for the ResonantNode class."""

    def test_energy_conservation_undamped(self):
        """
        Tests that an undamped, uncoupled node conserves energy.
        |signal| should remain constant.
        """
        node = ResonantNode(
            node_id="test_energy",
            frequency=1.0,
            amplitude=0.5,
            damping=0.0  # No damping
        )
        
        initial_energy = node.energy()
        initial_amplitude = node.get_amplitude()
        
        # Evolve the node for several steps
        for _ in range(100):
            node.step(dt=0.01)
            
        final_energy = node.energy()
        final_amplitude = node.get_amplitude()
        
        # The energy and amplitude should be almost identical to the initial values
        self.assertAlmostEqual(initial_energy, final_energy, places=7,
                             msg="Energy should be conserved for an undamped node.")
        self.assertAlmostEqual(initial_amplitude, final_amplitude, places=7,
                             msg="Amplitude should be constant for an undamped node.")

    def test_damping_reduces_energy(self):
        """
        Tests that a damped node loses energy over time.
        """
        node = ResonantNode(
            node_id="test_damping",
            frequency=1.0,
            amplitude=1.0,
            damping=0.1  # With damping
        )
        
        initial_energy = node.energy()
        
        # Evolve the node
        for _ in range(10):
            node.step(dt=0.1)
            
        final_energy = node.energy()
        
        self.assertLess(final_energy, initial_energy,
                        msg="Energy should decrease for a damped node.")

    def test_pickling_compatibility(self):
        """
        Tests that a ResonantNode can be pickled and unpickled correctly.
        """
        original_node = ResonantNode(
            node_id="pickle_test",
            frequency=5.5,
            amplitude=0.75,
            phase=np.pi / 2,
            damping=0.05,
            metadata={'custom': 'value'}
        )
        
        # Pickle the node
        pickled_data = pickle.dumps(original_node)
        
        # Unpickle the node
        unpickled_node = pickle.loads(pickled_data)
        
        # Assert that the state is restored correctly
        self.assertEqual(original_node.node_id, unpickled_node.node_id)
        self.assertEqual(original_node.metadata, unpickled_node.metadata)
        self.assertAlmostEqual(original_node.natural_freq, unpickled_node.natural_freq, places=7)
        self.assertAlmostEqual(original_node.damping, unpickled_node.damping, places=7)
        
        # Most importantly, check the complex signal
        self.assertAlmostEqual(original_node.signal.real, unpickled_node.signal.real, places=7)
        self.assertAlmostEqual(original_node.signal.imag, unpickled_node.signal.imag, places=7)
        
        # Also check convenience properties
        self.assertAlmostEqual(original_node.get_amplitude(), unpickled_node.get_amplitude(), places=7)
        self.assertAlmostEqual(original_node.get_phase(), unpickled_node.get_phase(), places=7)


if __name__ == '__main__':
    unittest.main() 