import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any, Callable
from collections import Counter
import time
import json

# Import performance backend for acceleration
try:
    import sys
    import os
    # Add parent directory to path to import synthnn
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from synthnn.performance import BackendManager, BackendType
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"Warning: SynthNN performance backends not available. Using NumPy fallback. Error: {e}")

# Type aliases
HarmonicSignature = List[Tuple[float, float, float]]
PrimaryExcitationSpectrum = Dict[float, float]

###############################################################################
#  Enhanced Oscillator Network with Backend Support
###############################################################################
class AcceleratedOscillatorNetwork:
    """Vectorised network with optional GPU/Metal acceleration."""
    
    def __init__(
        self,
        frequencies: np.ndarray,
        coupling_matrix: Optional[np.ndarray] = None,
        coupling_strength: float = 0.2,
        use_backend: bool = True
    ) -> None:
        self.N = len(frequencies)
        self.freq = frequencies.astype(float)
        self.phase = np.random.uniform(0, 2 * np.pi, self.N).astype(float)
        self.radius = np.zeros(self.N, dtype=float)
        self.coupling_strength = float(coupling_strength)
        self.time = 0.0
        
        if coupling_matrix is None:
            self.W = np.zeros((self.N, self.N), dtype=float)
        else:
            assert coupling_matrix.shape == (self.N, self.N)
            self.W = coupling_matrix.copy().astype(float)
        np.fill_diagonal(self.W, 0.0)
        
        # Initialize backend if available and requested
        self.backend = None
        self.use_backend = use_backend and BACKEND_AVAILABLE
        if self.use_backend:
            try:
                backend_manager = BackendManager()
                self.backend = backend_manager.get_backend()  # Auto-selects best backend
                print(f"Using {self.backend.backend_type.value} backend for acceleration")
                
                # Transfer data to device
                self._device_phase = self.backend.to_device(self.phase.astype(np.float32))
                self._device_radius = self.backend.to_device(self.radius.astype(np.float32))
                self._device_freq = self.backend.to_device(self.freq.astype(np.float32))
                self._device_W = self.backend.to_device(self.W.astype(np.float32))
            except Exception as e:
                print(f"Failed to initialize backend: {e}. Falling back to NumPy.")
                self.use_backend = False
                self.backend = None

    def inject(self, freqs: List[float], amps: List[float]):
        if len(freqs) == 0 or len(amps) == 0:
            return
        log_network_freqs = np.log2(self.freq + 1e-9)
        for f, a in zip(freqs, amps):
            if f <= 0: continue
            idx = np.argmin(np.abs(log_network_freqs - np.log2(f)))
            self.radius[idx] = np.clip(self.radius[idx] + float(a), 0.0, 5.0)
        
        # Sync to device if using backend
        if self.use_backend and self.backend:
            self._device_radius = self.backend.to_device(self.radius.astype(np.float32))

    def step(self, dt: float):
        if self.use_backend and self.backend:
            # Accelerated computation on device
            self._step_accelerated(dt)
        else:
            # Original NumPy implementation
            self._step_numpy(dt)
        
        self.time += dt
    
    def _step_numpy(self, dt: float):
        """Original NumPy implementation."""
        phase_diff_ji = self.phase[None, :] - self.phase[:, None]
        coupling_effects = self.W * np.sin(phase_diff_ji) 
        coupling = self.coupling_strength * np.sum(coupling_effects, axis=1)
        r = self.radius
        coupling_r = 0.1 * coupling
        coupling_phi = coupling / np.maximum(r, 1e-6)
        r_dot = (1 - r ** 2) * r + coupling_r
        phi_dot = 2 * np.pi * self.freq + coupling_phi
        self.radius = np.clip(r + dt * r_dot, 0.0, 5.0)
        self.phase = (self.phase + dt * phi_dot) % (2 * np.pi)
    
    def _step_accelerated(self, dt: float):
        """GPU/Metal accelerated implementation."""
        try:
            # Use backend's phase coupling operation
            coupling = self.backend.phase_coupling(
                self._device_phase,
                self._device_W,
                self.backend.to_device(np.ones((self.N, self.N), dtype=np.float32))  # Connection mask
            )
            
            # Scale coupling
            coupling_scaled = self.backend.multiply(
                coupling,
                self.backend.to_device(np.array(self.coupling_strength, dtype=np.float32))
            )
            
            # Get radius from device for calculations
            r_device = self._device_radius
            
            # Coupling effects on radius and phase
            coupling_r = self.backend.multiply(coupling_scaled, 
                                             self.backend.to_device(np.array(0.1, dtype=np.float32)))
            
            # r_dot = (1 - r^2) * r + coupling_r
            r_squared = self.backend.multiply(r_device, r_device)
            # Use add with negative instead of subtract
            neg_r_squared = self.backend.multiply(
                r_squared,
                self.backend.to_device(np.array(-1.0, dtype=np.float32))
            )
            one_minus_r2 = self.backend.add(
                self.backend.to_device(np.ones(self.N, dtype=np.float32)),
                neg_r_squared
            )
            r_natural = self.backend.multiply(one_minus_r2, r_device)
            r_dot = self.backend.add(r_natural, coupling_r)
            
            # Update radius
            r_new = self.backend.add(r_device, 
                                   self.backend.multiply(r_dot, 
                                                        self.backend.to_device(np.array(dt, dtype=np.float32))))
            
            # Natural frequency advance
            freq_advance = self.backend.multiply(
                self._device_freq,
                self.backend.to_device(np.array(2 * np.pi * dt, dtype=np.float32))
            )
            
            # Total phase change
            phase_change = self.backend.add(freq_advance, 
                                          self.backend.multiply(coupling_scaled, 
                                                              self.backend.to_device(np.array(dt, dtype=np.float32))))
            
            # Update phase
            self._device_phase = self.backend.add(self._device_phase, phase_change)
            self._device_radius = r_new
            
            # Sync back to host
            self.phase = self.backend.to_host(self._device_phase) % (2 * np.pi)
            self.radius = np.clip(self.backend.to_host(self._device_radius), 0.0, 5.0)
            
        except Exception as e:
            print(f"Acceleration failed: {e}. Falling back to NumPy.")
            self._step_numpy(dt)

    def active_indices(self, thresh_ratio: float = 0.2) -> np.ndarray:
        mx = np.max(self.radius)
        if mx == 0:
            return np.array([], dtype=int)
        return np.where(self.radius > thresh_ratio * mx)[0]

    def signature(self, k: int = 12, thresh_ratio: float = 0.2) -> HarmonicSignature:
        idx = self.active_indices(thresh_ratio)
        if idx.size == 0: return []
        comps = [(float(self.freq[i]), float(self.radius[i]), float(self.phase[i])) for i in idx]
        comps.sort(key=lambda x: x[0])
        return comps[:k]

###############################################################################
#  Pattern Storage and Retrieval System (Domain-Neutral)
###############################################################################
class PatternMemory:
    """Generic pattern storage and retrieval system for harmonic signatures."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.patterns: List[Dict[str, Any]] = []
        self.pattern_index: Dict[str, List[int]] = {}  # Tag-based indexing
        
    def store_pattern(self, signature: HarmonicSignature, 
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None) -> str:
        """Store a harmonic signature with optional metadata and tags."""
        pattern_id = f"pattern_{len(self.patterns)}_{int(time.time() * 1000)}"
        
        pattern_data = {
            'id': pattern_id,
            'signature': signature,
            'metadata': metadata or {},
            'tags': tags or [],
            'timestamp': time.time(),
            'encoding': self._encode_signature(signature)
        }
        
        # Add to storage
        self.patterns.append(pattern_data)
        
        # Update indices
        for tag in pattern_data['tags']:
            if tag not in self.pattern_index:
                self.pattern_index[tag] = []
            self.pattern_index[tag].append(len(self.patterns) - 1)
        
        # Manage capacity
        if len(self.patterns) > self.capacity:
            self._evict_oldest()
            
        return pattern_id
    
    def retrieve_pattern(self, pattern_id: str) -> Optional[HarmonicSignature]:
        """Retrieve a pattern by ID."""
        for pattern in self.patterns:
            if pattern['id'] == pattern_id:
                return pattern['signature']
        return None
    
    def find_similar_patterns(self, query_signature: HarmonicSignature, 
                            threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find patterns similar to the query signature."""
        query_encoding = self._encode_signature(query_signature)
        results = []
        
        for pattern in self.patterns:
            similarity = self._compute_similarity(query_encoding, pattern['encoding'])
            if similarity >= threshold:
                results.append((pattern['id'], similarity))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _encode_signature(self, signature: HarmonicSignature) -> np.ndarray:
        """Encode signature as a fixed-size vector for comparison."""
        if not signature:
            return np.zeros(128)
            
        # Simple encoding: histogram of frequencies and amplitudes
        freqs = [s[0] for s in signature]
        amps = [s[1] for s in signature]
        
        # Create frequency histogram (64 bins)
        freq_hist, _ = np.histogram(freqs, bins=64, range=(0, 10000))
        freq_hist = freq_hist / (np.sum(freq_hist) + 1e-8)
        
        # Create amplitude histogram (64 bins)
        amp_hist, _ = np.histogram(amps, bins=64, range=(0, 5))
        amp_hist = amp_hist / (np.sum(amp_hist) + 1e-8)
        
        return np.concatenate([freq_hist, amp_hist])
    
    def _compute_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Compute similarity between two encodings."""
        # Cosine similarity
        dot_product = np.dot(encoding1, encoding2)
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _evict_oldest(self):
        """Remove oldest pattern when capacity is exceeded."""
        if self.patterns:
            removed = self.patterns.pop(0)
            # Update indices
            for tag in removed['tags']:
                self.pattern_index[tag] = [idx - 1 for idx in self.pattern_index[tag] if idx > 0]

###############################################################################
#  Enhanced DHRC Layer with Acceleration and Memory
###############################################################################
class DHRCNetworkLayerEnhanced:
    def __init__(
        self,
        name: str,
        num_nodes: int,
        freq_range: Tuple[float, float],
        fixed_frequencies: Optional[np.ndarray] = None,
        connection_prob: float = 0.3,
        coupling_strength_within_layer: float = 0.15,
        learning_rate_W: float = 0.001,
        weight_decay_W: float = 0.0001,
        max_abs_weight_W: float = 1.0,
        use_acceleration: bool = True,
        enable_memory: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.name = name
        self.rng = rng if rng is not None else np.random.default_rng(hash(name) % (2**32 -1))
        self.learning_rate_W = learning_rate_W
        self.weight_decay_W = weight_decay_W
        self.max_abs_weight_W = max_abs_weight_W
        self.use_acceleration = use_acceleration
        
        # Pattern memory for this layer
        self.memory = PatternMemory(capacity=100) if enable_memory else None
        
        if fixed_frequencies is not None:
            assert len(fixed_frequencies) == num_nodes
            freqs = np.asarray(fixed_frequencies, dtype=float)
        else:
            base_log_freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), num_nodes)
            jitter_factor = 0.02
            freq_jitters = 1 + self.rng.uniform(-jitter_factor, jitter_factor, num_nodes)
            freqs = np.clip(base_log_freqs * freq_jitters, freq_range[0], freq_range[1])
            freqs = np.sort(freqs)

        # Initialize connection matrix
        W = np.zeros((num_nodes, num_nodes), dtype=float)
        lf = np.log2(freqs + 1e-9)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                log_dist_sq = (lf[i] - lf[j]) ** 2
                prob_factor = np.exp(-(log_dist_sq * 6))
                if self.rng.random() < (connection_prob * prob_factor):
                    W[i, j] = self.rng.normal(loc=0.0, scale=0.2)
        
        W = np.clip(W, -self.max_abs_weight_W, self.max_abs_weight_W)
        
        # Use accelerated network if available
        self.net = AcceleratedOscillatorNetwork(
            freqs, W, 
            coupling_strength=coupling_strength_within_layer,
            use_backend=use_acceleration
        )

        self.harmonic_ratios = np.array([
            0.25, 0.333, 0.5, 2/3, 3/4, 4/5, 5/6, 1.0, 6/5, 5/4, 4/3, 3/2, 5/3, 8/5, 7/4, 9/5, 2.0, 2.5, 3.0, 4.0
        ])
        self.ratio_tol = 0.04

    def _is_harmonic(self, freq: float, fundamental: float) -> bool:
        if fundamental <= 1e-6: return False
        r = freq / fundamental
        for hr in self.harmonic_ratios:
            if abs(r - hr) < self.ratio_tol * hr: return True
            if r > 1e-6 and abs((1.0/r) - hr) < self.ratio_tol * hr: return True
        return False

    def _hebbian_update_W(self, dt: float):
        """Updates connection matrix W based on Hebbian learning principles."""
        r = self.net.radius
        phi = self.net.phase

        co_activation = np.outer(r, r)
        phase_diff_ji = phi[None, :] - phi[:, None]
        phase_coherence = np.cos(phase_diff_ji)

        delta_W = self.learning_rate_W * co_activation * phase_coherence * dt
        
        self.net.W += delta_W
        self.net.W *= (1.0 - self.weight_decay_W * dt)
        self.net.W = np.clip(self.net.W, -self.max_abs_weight_W, self.max_abs_weight_W)
        np.fill_diagonal(self.net.W, 0.0)
        
        # Update device copy if using acceleration
        if self.net.use_backend and self.net.backend:
            self.net._device_W = self.net.backend.to_device(self.net.W.astype(np.float32))

    def process(
        self,
        input_spec: Union[PrimaryExcitationSpectrum, HarmonicSignature],
        iters: int = 50,
        steps_per_iter: int = 10,
        dt: float = 0.01,
        harmonic_boost: float = 1.02,
        dissonant_damping: float = 0.99,
        active_thresh_ratio_for_fundamental: float = 0.05,
        active_thresh_ratio_for_output: float = 0.10,
        k_signature: int = 15,
        verbose: bool = False,
        store_output: bool = True
    ) -> HarmonicSignature:
        if verbose: print(f"\n--- Processing Layer (Enhanced): {self.name} ---")

        # Reset network state
        self.net.phase = self.rng.uniform(0, 2 * np.pi, self.net.N).astype(float)
        self.net.radius.fill(0.0)
        self.net.time = 0.0

        # Inject input
        if isinstance(input_spec, dict):
            if not input_spec:
                 if verbose: print(f"Layer {self.name}: Empty PES. Skipping.")
                 return []
            freqs, amps = zip(*input_spec.items())
            self.net.inject(list(freqs), list(amps))
        else:
            if not input_spec:
                if verbose: print(f"Layer {self.name}: Empty HarmonicSignature input. Skipping.")
                return []
            freqs, amps, _ = zip(*input_spec)
            self.net.inject(list(freqs), list(amps))
        
        # Check if we've seen similar input before
        if self.memory and isinstance(input_spec, list):
            similar_patterns = self.memory.find_similar_patterns(input_spec, threshold=0.9)
            if similar_patterns and verbose:
                print(f"  Found {len(similar_patterns)} similar patterns in memory")

        # Processing loop
        for t_iter in range(iters):
            idx_active = self.net.active_indices(active_thresh_ratio_for_fundamental)
            
            fundamental = 0.0
            if len(idx_active) > 0:
                active_freqs = self.net.freq[idx_active]
                active_radii = self.net.radius[idx_active]
                
                if np.sum(active_radii) > 1e-6:
                    sorted_indices = np.argsort(active_freqs)
                    sorted_freqs = active_freqs[sorted_indices]
                    sorted_radii_weights = active_radii[sorted_indices]
                    cumulative_weights = np.cumsum(sorted_radii_weights)
                    median_weight_idx = np.searchsorted(cumulative_weights, cumulative_weights[-1] / 2.0)
                    if median_weight_idx >= len(sorted_freqs): median_weight_idx = len(sorted_freqs)-1
                    fundamental = float(sorted_freqs[median_weight_idx])
                else:
                    fundamental = float(np.median(active_freqs)) if len(active_freqs)>0 else 0.0

            if verbose and (t_iter < 2 or (t_iter+1)%10==0):
                print(f"  Iter {t_iter+1}: Candidate Fundamental = {fundamental:.2f} Hz ({len(idx_active)} active for fund.)")

            # Apply harmonic/dissonant adjustments
            if fundamental > 1e-6:
                for i in range(self.net.N):
                    if self._is_harmonic(self.net.freq[i], fundamental):
                        self.net.radius[i] *= harmonic_boost
                    else:
                        self.net.radius[i] *= dissonant_damping
            
            # Energy normalization
            current_total_energy = np.sum(self.net.radius**2)
            num_contributing_nodes_approx = max(1, len(idx_active) if len(idx_active) > 0 else self.net.N // 2)

            if current_total_energy > 1e-9:
                 norm_factor = np.sqrt(num_contributing_nodes_approx / current_total_energy)
                 self.net.radius *= norm_factor
            
            self.net.radius = np.clip(self.net.radius, 0.0, 5.0)

            # Dynamic steps
            for step_num in range(steps_per_iter):
                self.net.step(dt)
                if step_num == steps_per_iter -1:
                     self._hebbian_update_W(dt)

        # Generate output signature
        final_signature = self.net.signature(k=k_signature, thresh_ratio=active_thresh_ratio_for_output)
        
        # Store in memory if enabled
        if self.memory and store_output and final_signature:
            pattern_id = self.memory.store_pattern(
                final_signature,
                metadata={'layer': self.name, 'fundamental': fundamental},
                tags=[self.name, f'fund_{int(fundamental)}']
            )
            if verbose:
                print(f"  Stored pattern {pattern_id} in memory")
        
        if verbose:
            print(f"Layer {self.name} processed. Output Signature ({len(final_signature)} components):")
            for comp_idx, (f, a, p) in enumerate(final_signature[:5]): 
                print(f"  Comp {comp_idx}: Freq={f:.2f} Hz, Amp={a:.3f}, Phase={p:.2f} rad")
            if len(final_signature) > 5: 
                print(f"  ...and {len(final_signature)-5} more.")
            elif not final_signature: 
                print("  (empty signature)")
                
        return final_signature

###############################################################################
#  Evolutionary Parameter Optimization (Domain-Neutral)
###############################################################################
class CascadeEvolution:
    """Evolve cascade parameters for optimal performance."""
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_params = None
        
    def initialize_population(self, param_ranges: Dict[str, Tuple[float, float]]):
        """Initialize population with random parameters."""
        self.param_ranges = param_ranges
        self.population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in param_ranges.items():
                individual[param] = np.random.uniform(min_val, max_val)
            self.population.append(individual)
    
    def evaluate_fitness(self, params: Dict[str, float], 
                        fitness_function: Callable) -> float:
        """Evaluate fitness of a parameter set."""
        return fitness_function(params)
    
    def evolve_generation(self, fitness_function: Callable):
        """Evolve one generation."""
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in self.population:
            fitness = self.evaluate_fitness(individual, fitness_function)
            fitness_scores.append(fitness)
            
            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_params = individual.copy()
        
        # Selection (tournament)
        new_population = []
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(self.population_size, tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            new_population.append(self.population[winner_idx].copy())
        
        # Crossover and mutation
        for i in range(0, self.population_size - 1, 2):
            if np.random.random() < self.crossover_rate:
                # Uniform crossover
                for param in self.param_ranges:
                    if np.random.random() < 0.5:
                        new_population[i][param], new_population[i+1][param] = \
                            new_population[i+1][param], new_population[i][param]
        
        # Mutation
        for individual in new_population:
            for param, (min_val, max_val) in self.param_ranges.items():
                if np.random.random() < self.mutation_rate:
                    # Gaussian mutation
                    mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                    individual[param] = np.clip(individual[param] + mutation, min_val, max_val)
        
        self.population = new_population
        self.generation += 1
        
        return self.best_fitness, self.best_params

###############################################################################
#  Enhanced DHRC System
###############################################################################
class DHRCSystemEnhanced:
    def __init__(self, 
                 layer_cfgs: List[Dict[str, Any]], 
                 global_seed: Optional[int] = None,
                 use_acceleration: bool = True,
                 enable_memory: bool = True):
        self.layer_cfgs = layer_cfgs
        self.layers: List[DHRCNetworkLayerEnhanced] = []
        self.use_acceleration = use_acceleration
        self.enable_memory = enable_memory
        
        # Global pattern memory
        self.global_memory = PatternMemory(capacity=1000) if enable_memory else None
        
        # Performance metrics
        self.processing_times = []
        
        master_rng = np.random.default_rng(global_seed)
        
        for i, cfg in enumerate(layer_cfgs):
            layer_seed = master_rng.integers(2**32 - 1) if global_seed is not None else None
            layer_rng = np.random.default_rng(layer_seed)
            self.layers.append(
                DHRCNetworkLayerEnhanced(
                    name=cfg.get("name", f"L{i+1}"),
                    num_nodes=cfg["num_nodes"],
                    freq_range=cfg["freq_range"],
                    fixed_frequencies=cfg.get("fixed_node_frequencies"),
                    connection_prob=cfg.get("connection_prob", 0.25),
                    coupling_strength_within_layer=cfg.get("coupling_strength", 0.1),
                    learning_rate_W=cfg.get("learning_rate_W", 0.001),
                    weight_decay_W=cfg.get("weight_decay_W", 0.0001),
                    max_abs_weight_W=cfg.get("max_abs_weight_W", 0.75),
                    use_acceleration=use_acceleration,
                    enable_memory=enable_memory,
                    rng=layer_rng
                )
            )

    def run(self, pes: PrimaryExcitationSpectrum, 
            verbose_per_layer: bool = False,
            measure_performance: bool = True) -> List[HarmonicSignature]:
        current_spectrum: Union[PrimaryExcitationSpectrum, HarmonicSignature] = pes
        all_layer_signatures: List[HarmonicSignature] = []
        
        start_time = time.time()
        
        print(f"\n=== DHRC System (Enhanced): Starting Run ===")
        print(f"Initial PES ({len(pes)} components):")
        for f_idx, (f_val, a_val) in enumerate(list(pes.items())[:3]):
             print(f"  PES Comp {f_idx}: Freq={f_val:.2f}, Amp={a_val:.3f}")
        if len(pes)>3: print("  ...")

        layer_times = []
        
        for i, layer in enumerate(self.layers):
            layer_start = time.time()
            
            output_signature = layer.process(
                current_spectrum, 
                iters=self.layer_cfgs[i].get("iters", 50),
                steps_per_iter=self.layer_cfgs[i].get("steps_per_iter", 10),
                verbose=verbose_per_layer
            )
            
            layer_time = time.time() - layer_start
            layer_times.append(layer_time)
            
            all_layer_signatures.append(output_signature)
            
            if not output_signature:
                print(f"Layer {layer.name} (Layer {i+1}) produced an empty signature. Halting cascade.")
                for _ in range(i + 1, len(self.layers)): 
                    all_layer_signatures.append([])
                break
                
            current_spectrum = output_signature

        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        
        if measure_performance:
            print(f"\n=== Performance Metrics ===")
            print(f"Total processing time: {total_time:.3f}s")
            for i, (layer, layer_time) in enumerate(zip(self.layers, layer_times)):
                print(f"  Layer {i+1} ({layer.name}): {layer_time:.3f}s")
        
        # Store final output in global memory
        if self.global_memory and all_layer_signatures and all_layer_signatures[-1]:
            self.global_memory.store_pattern(
                all_layer_signatures[-1],
                metadata={'cascade_depth': len(self.layers), 'processing_time': total_time},
                tags=['cascade_output', f'depth_{len(self.layers)}']
            )
        
        print(f"\n=== DHRC System (Enhanced): Run Complete ===")
        return all_layer_signatures
    
    def optimize_parameters(self, 
                          test_inputs: List[PrimaryExcitationSpectrum],
                          fitness_metric: str = 'convergence_speed',
                          generations: int = 20) -> Dict[str, float]:
        """Use evolutionary optimization to find optimal cascade parameters."""
        
        # Define parameter ranges to optimize
        param_ranges = {
            'coupling_strength': (0.05, 0.5),
            'learning_rate_W': (0.0001, 0.01),
            'weight_decay_W': (0.00001, 0.001),
            'harmonic_boost': (1.01, 1.1),
            'dissonant_damping': (0.9, 0.99)
        }
        
        evolution = CascadeEvolution(population_size=20)
        evolution.initialize_population(param_ranges)
        
        def fitness_function(params: Dict[str, float]) -> float:
            """Evaluate cascade performance with given parameters."""
            # Create temporary cascade with test parameters
            test_layer_cfgs = []
            for cfg in self.layer_cfgs:
                test_cfg = cfg.copy()
                test_cfg['coupling_strength'] = params['coupling_strength']
                test_cfg['learning_rate_W'] = params['learning_rate_W']
                test_cfg['weight_decay_W'] = params['weight_decay_W']
                test_layer_cfgs.append(test_cfg)
            
            test_cascade = DHRCSystemEnhanced(
                test_layer_cfgs, 
                use_acceleration=self.use_acceleration,
                enable_memory=False  # Disable memory for testing
            )
            
            # Evaluate on test inputs
            fitness_scores = []
            for test_input in test_inputs:
                signatures = test_cascade.run(test_input, verbose_per_layer=False, measure_performance=False)
                
                if fitness_metric == 'convergence_speed':
                    # Measure how quickly the cascade converges to stable output
                    if signatures and signatures[-1]:
                        fitness = len(signatures[-1]) / test_cascade.processing_times[-1]
                    else:
                        fitness = 0
                elif fitness_metric == 'output_richness':
                    # Measure richness of final output
                    if signatures and signatures[-1]:
                        fitness = len(signatures[-1]) * np.mean([s[1] for s in signatures[-1]])
                    else:
                        fitness = 0
                else:
                    fitness = 0
                    
                fitness_scores.append(fitness)
            
            return np.mean(fitness_scores)
        
        # Run evolution
        print(f"\n=== Optimizing Cascade Parameters ===")
        for gen in range(generations):
            best_fitness, best_params = evolution.evolve_generation(fitness_function)
            if gen % 5 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness:.4f}")
        
        print(f"\nOptimal parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.6f}")
            
        return best_params

###############################################################################
#  Demo with enhanced features
###############################################################################
if __name__ == "__main__":
    example_pes = {
        100.0: 0.9, 149.5: 0.7, 205.0: 0.6, 220.0: 0.35, 310.0: 0.5, 440.0: 0.25
    }

    layer_configurations_enhanced = [
        {
            "name": "DetectorLayer (L1)", "num_nodes": 40, "freq_range": (30, 700),
            "connection_prob": 0.3, "coupling_strength": 0.18, 
            "learning_rate_W": 0.0005, "weight_decay_W": 0.00005, "max_abs_weight_W": 0.6,
            "iters": 60, "steps_per_iter": 12
        },
        {
            "name": "RelationLayer (L2)", "num_nodes": 30, "freq_range": (50, 1500),
            "connection_prob": 0.25, "coupling_strength": 0.15,
            "learning_rate_W": 0.0003, "weight_decay_W": 0.00003, "max_abs_weight_W": 0.5,
             "iters": 50, "steps_per_iter": 10
        },
        {
            "name": "AbstractHarmonyLayer (L3)", "num_nodes": 20, "freq_range": (1.0, 60.0),
            "fixed_node_frequencies": np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 48, 60
            ]) * 2.5,
            "connection_prob": 0.2, "coupling_strength": 0.12,
            "learning_rate_W": 0.0001, "weight_decay_W": 0.00001, "max_abs_weight_W": 0.4,
             "iters": 40, "steps_per_iter": 8
        },
    ]

    # Test with acceleration
    print("\n=== Testing Enhanced Cascade with Acceleration ===")
    dhrc_system_enhanced = DHRCSystemEnhanced(
        layer_configurations_enhanced, 
        global_seed=123,
        use_acceleration=False,
        enable_memory=True
    )
    
    final_signatures_enhanced = dhrc_system_enhanced.run(example_pes, verbose_per_layer=False)

    print("\n--- Final Harmonic Signatures (Enhanced) ---")
    for i, sig in enumerate(final_signatures_enhanced):
        layer_name = dhrc_system_enhanced.layers[i].name
        print(f"\nLayer {i+1} ({layer_name}) Signature:")
        if sig:
            for comp_idx, (freq, amp, phase) in enumerate(sig[:10]):
                print(f"  Component {comp_idx}: Freq={freq:.2f} Hz, Amp={amp:.4f}, Phase={phase:.2f} rad")
            if len(sig) > 10: print(f"  ... and {len(sig)-10} more components.")
        else: print("  (empty signature)")

    # Test pattern memory
    if dhrc_system_enhanced.global_memory:
        print("\n--- Pattern Memory Test ---")
        # Find similar patterns to the output
        if final_signatures_enhanced and final_signatures_enhanced[-1]:
            similar = dhrc_system_enhanced.global_memory.find_similar_patterns(
                final_signatures_enhanced[-1], 
                threshold=0.8
            )
            print(f"Found {len(similar)} similar patterns in memory")
    
    # Test parameter optimization (with small test set for demo)
    print("\n--- Parameter Optimization Test ---")
    test_inputs = [example_pes, {200.0: 0.8, 300.0: 0.6, 400.0: 0.4}]
    # Note: This would take time with many generations, so keeping it small for demo
    # optimal_params = dhrc_system_enhanced.optimize_parameters(
    #     test_inputs, 
    #     fitness_metric='convergence_speed',
    #     generations=5
    # )