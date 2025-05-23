from abstract import ResonantLayer

class HierarchicalResonantNetwork:
    def __init__(self, layers):
        self.layers = []
        for i, (num_nodes, base_freq) in enumerate(layers):
            # Each layer operates at different frequency scales
            layer = ResonantLayer(
                num_nodes=num_nodes,
                base_freq=base_freq * (2 ** i),  # Octave relationships
                harmonic_structure='adaptive'
            )
            self.layers.append(layer)
    
    def forward(self, input_signal):
        # Progressive refinement through frequency scales
        for layer in self.layers:
            input_signal = layer.resonate(input_signal)
        return input_signal