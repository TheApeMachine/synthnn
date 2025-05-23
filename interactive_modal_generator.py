import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
from modal_music_generator import ModalMusicGenerator, ModalPhraseGenerator, RhythmGenerator
from context_aware_detector import ContextAwareModeDetector

class InteractiveModalGenerator:
    """
    Interactive music generator that responds to user input and generates
    music in real-time using modal networks.
    """
    
    def __init__(self, base_freq=440.0, sample_rate=44100):
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        
        # Core generator
        self.generator = ModalMusicGenerator(base_freq, sample_rate)
        
        # Real-time parameters
        self.current_params = {
            'mode': 'Ionian',
            'tempo': 120,
            'complexity': 0.5,
            'brightness': 0.7,
            'density': 0.5,
            'variation': 0.3
        }
        
        # Generation state
        self.is_generating = False
        self.generation_thread = None
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Interaction history
        self.interaction_history = []
        self.parameter_curves = {param: [] for param in self.current_params}
        
        # Mode transition system
        self.mode_affinity_graph = self._build_mode_affinity_graph()
        
    def _build_mode_affinity_graph(self):
        """Build a graph of mode relationships for smooth transitions"""
        return {
            'Ionian': {'weight': 1.0, 'neighbors': ['Lydian', 'Mixolydian', 'Aeolian']},
            'Dorian': {'weight': 0.8, 'neighbors': ['Aeolian', 'Mixolydian', 'Phrygian']},
            'Phrygian': {'weight': 0.6, 'neighbors': ['Dorian', 'Aeolian', 'Locrian']},
            'Lydian': {'weight': 1.0, 'neighbors': ['Ionian', 'Mixolydian']},
            'Mixolydian': {'weight': 0.9, 'neighbors': ['Ionian', 'Dorian', 'Lydian']},
            'Aeolian': {'weight': 0.7, 'neighbors': ['Dorian', 'Phrygian', 'Ionian']},
            'Locrian': {'weight': 0.4, 'neighbors': ['Phrygian']}
        }
    
    def update_parameter(self, param_name, value):
        """Update a generation parameter in real-time"""
        if param_name in self.current_params:
            old_value = self.current_params[param_name]
            self.current_params[param_name] = value
            
            # Record parameter change
            self.parameter_curves[param_name].append({
                'time': time.time(),
                'value': value
            })
            
            # Handle special parameters
            if param_name == 'mode':
                self._handle_mode_transition(old_value, value)
            elif param_name == 'tempo':
                self.generator.tempo = value
                self.generator.beat_duration = 60.0 / value
            
            print(f"Updated {param_name}: {old_value} -> {value}")
    
    def _handle_mode_transition(self, from_mode, to_mode):
        """Handle smooth mode transitions"""
        # Find transition path
        if from_mode != to_mode:
            path = self.generator.context_detector.suggest_mode_transition(
                from_mode, 
                self._create_transition_signal(to_mode),
                self.sample_rate
            )
            
            # Schedule gradual transition
            self.scheduled_transitions = path[1:]  # Skip current mode
            print(f"Mode transition path: {' -> '.join(path)}")
    
    def _create_transition_signal(self, target_mode):
        """Create a signal that represents the target mode"""
        t = np.linspace(0, 0.5, int(0.5 * self.sample_rate))
        intervals = self.generator.context_detector.mode_intervals[target_mode]
        
        signal = np.zeros_like(t)
        for i, interval in enumerate(intervals[:3]):
            freq = self.base_freq * interval
            signal += (0.5 / (i + 1)) * np.sin(2 * np.pi * freq * t)
        
        return signal
    
    def generate_interactive_phrase(self):
        """Generate a single phrase based on current parameters"""
        # Determine semantic tags from parameters
        semantic_tags = []
        
        if self.current_params['brightness'] > 0.7:
            semantic_tags.append('bright')
        elif self.current_params['brightness'] < 0.3:
            semantic_tags.append('dark')
            
        if self.current_params['complexity'] > 0.7:
            semantic_tags.append('complex')
        elif self.current_params['complexity'] < 0.3:
            semantic_tags.append('simple')
            
        if self.current_params['variation'] > 0.7:
            semantic_tags.append('experimental')
            
        # Generate phrase with current settings
        duration = 4 * self.generator.beat_duration  # 4-beat phrase
        
        # Use parameter-driven generation
        phrase_audio = self._generate_parametric_phrase(
            duration,
            self.current_params,
            semantic_tags
        )
        
        return phrase_audio
    
    def _generate_parametric_phrase(self, duration, params, semantic_tags):
        """Generate a phrase with specific parameters"""
        # Create melodic contour influenced by parameters
        contour_generator = ParametricContourGenerator(params)
        melody_contour = contour_generator.generate(
            self.current_params['mode'],
            int(duration * 4),  # Quarter note resolution
            params['variation']
        )
        
        # Create rhythm pattern based on density
        rhythm_generator = ParametricRhythmGenerator()
        rhythm_pattern = rhythm_generator.generate(
            duration,
            params['density'],
            params['complexity']
        )
        
        # Render with brightness-adjusted harmonics
        phrase_audio = self._render_parametric_phrase(
            melody_contour,
            rhythm_pattern,
            params
        )
        
        return phrase_audio
    
    def _render_parametric_phrase(self, melody_contour, rhythm_pattern, params):
        """Render phrase with parameter-controlled synthesis"""
        samples = int(sum(rhythm_pattern) * self.generator.beat_duration * self.sample_rate)
        audio = np.zeros(samples)
        
        mode_intervals = self.generator.context_detector.mode_intervals[params['mode']]
        
        current_sample = 0
        for note_degree, duration_beats in zip(melody_contour, rhythm_pattern):
            note_duration = duration_beats * self.generator.beat_duration
            note_samples = int(note_duration * self.sample_rate)
            
            if note_degree > 0:
                # Calculate frequency
                if note_degree <= len(mode_intervals):
                    freq_ratio = mode_intervals[note_degree - 1]
                else:
                    octave = (note_degree - 1) // len(mode_intervals)
                    degree_in_octave = (note_degree - 1) % len(mode_intervals)
                    freq_ratio = mode_intervals[degree_in_octave] * (2 ** octave)
                
                note_freq = self.base_freq * freq_ratio
                
                # Generate with brightness-controlled harmonics
                t = np.linspace(0, note_duration, note_samples)
                note_audio = self._synthesize_note(
                    t, note_freq, params['brightness'], params['mode']
                )
                
                # Apply envelope
                envelope = self._create_parametric_envelope(
                    note_samples, 
                    note_duration,
                    params['complexity']
                )
                note_audio *= envelope
                
                # Add to output
                end_sample = min(current_sample + note_samples, samples)
                audio[current_sample:end_sample] = note_audio[:end_sample-current_sample]
            
            current_sample += note_samples
            if current_sample >= samples:
                break
        
        return audio
    
    def _synthesize_note(self, t, freq, brightness, mode):
        """Synthesize a note with brightness control"""
        note = np.zeros_like(t)
        
        # Fundamental
        note += 0.6 * np.sin(2 * np.pi * freq * t)
        
        # Brightness-controlled harmonics
        num_harmonics = int(2 + brightness * 8)  # 2-10 harmonics
        
        for h in range(2, num_harmonics + 1):
            # Adjust harmonic amplitude based on mode character
            if mode in ['Lydian', 'Ionian']:
                amp = 0.3 / h * brightness
            elif mode in ['Phrygian', 'Locrian']:
                amp = 0.2 / h * (1 - brightness * 0.5)
            else:
                amp = 0.25 / h
            
            note += amp * np.sin(2 * np.pi * freq * h * t)
        
        return note
    
    def _create_parametric_envelope(self, num_samples, duration, complexity):
        """Create envelope with complexity-based shape"""
        if complexity < 0.3:
            # Simple envelope
            attack = 0.05
            decay = 0.1
            sustain = 0.8
            release = 0.15
        elif complexity > 0.7:
            # Complex envelope
            attack = 0.01
            decay = 0.2
            sustain = 0.5
            release = 0.3
        else:
            # Medium envelope
            attack = 0.02
            decay = 0.15
            sustain = 0.7
            release = 0.2
        
        attack_samples = int(attack * duration * self.sample_rate)
        decay_samples = int(decay * duration * self.sample_rate)
        release_samples = int(release * duration * self.sample_rate)
        sustain_samples = num_samples - attack_samples - decay_samples - release_samples
        
        envelope = np.concatenate([
            np.linspace(0, 1, attack_samples),
            np.linspace(1, sustain, decay_samples),
            np.ones(max(0, sustain_samples)) * sustain,
            np.linspace(sustain, 0, release_samples)
        ])
        
        return envelope[:num_samples]
    
    def start_generation(self):
        """Start the real-time generation thread"""
        if not self.is_generating:
            self.is_generating = True
            self.generation_thread = threading.Thread(target=self._generation_loop)
            self.generation_thread.start()
            print("Started real-time generation")
    
    def stop_generation(self):
        """Stop the generation thread"""
        self.is_generating = False
        if self.generation_thread:
            self.generation_thread.join()
        print("Stopped generation")
    
    def _generation_loop(self):
        """Main generation loop running in separate thread"""
        while self.is_generating:
            # Generate new phrase
            phrase = self.generate_interactive_phrase()
            
            # Add to queue if space available
            try:
                self.audio_queue.put(phrase, block=False)
            except queue.Full:
                pass  # Skip if queue is full
            
            # Small delay to control generation rate
            time.sleep(0.1)
    
    def get_visualization_data(self):
        """Get data for real-time visualization"""
        return {
            'mode': self.current_params['mode'],
            'parameter_history': self.parameter_curves,
            'queue_size': self.audio_queue.qsize(),
            'mode_graph': self.mode_affinity_graph
        }


class ParametricContourGenerator:
    """Generate melodic contours based on parameters"""
    
    def __init__(self, params):
        self.params = params
        
    def generate(self, mode, length, variation):
        """Generate contour with controlled variation"""
        # Base patterns for each parameter level
        if variation < 0.3:
            # Conservative melodic movement
            return self._generate_conservative(mode, length)
        elif variation > 0.7:
            # Experimental melodic movement
            return self._generate_experimental(mode, length)
        else:
            # Balanced melodic movement
            return self._generate_balanced(mode, length)
    
    def _generate_conservative(self, mode, length):
        """Generate conservative melodic patterns"""
        # Mostly stepwise motion, predictable patterns
        contour = [1]  # Start on tonic
        
        for i in range(1, length):
            if np.random.random() < 0.8:
                # Stepwise motion
                step = np.random.choice([-1, 0, 1], p=[0.3, 0.2, 0.5])
                next_degree = max(1, min(8, contour[-1] + step))
            else:
                # Occasional jump to chord tone
                next_degree = np.random.choice([1, 3, 5])
            
            contour.append(next_degree)
        
        return contour
    
    def _generate_experimental(self, mode, length):
        """Generate experimental melodic patterns"""
        # Wide intervals, unexpected movements
        contour = [np.random.randint(1, 8)]
        
        for i in range(1, length):
            if np.random.random() < 0.6:
                # Large interval
                jump = np.random.choice([-5, -4, -3, 3, 4, 5])
                next_degree = max(1, min(15, contour[-1] + jump))
            else:
                # Chromatic or unexpected movement
                next_degree = np.random.randint(1, 12)
            
            contour.append(next_degree)
        
        return contour
    
    def _generate_balanced(self, mode, length):
        """Generate balanced melodic patterns"""
        # Mix of stepwise and leaps
        contour = [np.random.choice([1, 3, 5])]
        
        for i in range(1, length):
            if np.random.random() < 0.6:
                # Stepwise
                step = np.random.choice([-2, -1, 1, 2])
                next_degree = max(1, min(8, contour[-1] + step))
            else:
                # Leap
                leap = np.random.choice([-4, -3, 3, 4])
                next_degree = max(1, min(10, contour[-1] + leap))
            
            contour.append(next_degree)
        
        return contour


class ParametricRhythmGenerator:
    """Generate rhythm patterns based on parameters"""
    
    def generate(self, duration, density, complexity):
        """Generate rhythm with controlled density and complexity"""
        beat_duration = 0.5  # Assume 120 BPM
        total_beats = duration / beat_duration
        
        if density < 0.3:
            # Sparse rhythm
            base_values = [1, 2, 2, 4]
        elif density > 0.7:
            # Dense rhythm
            base_values = [0.25, 0.5, 0.5, 0.25]
        else:
            # Medium density
            base_values = [0.5, 1, 1, 0.5]
        
        # Add complexity through variation
        pattern = []
        current_total = 0
        
        while current_total < total_beats:
            if complexity > 0.5 and np.random.random() < complexity:
                # Add syncopation or complex pattern
                value = np.random.choice([0.25, 0.5, 0.75, 1.5])
            else:
                value = np.random.choice(base_values)
            
            if current_total + value <= total_beats:
                pattern.append(value)
                current_total += value
            else:
                # Fill remaining time
                pattern.append(total_beats - current_total)
                break
        
        return pattern


def create_interactive_interface():
    """Create an interactive GUI for the generator"""
    generator = InteractiveModalGenerator()
    
    # Create matplotlib figure for visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Parameter sliders
    from matplotlib.widgets import Slider, Button, RadioButtons
    
    # Create slider axes
    ax_tempo = plt.axes([0.1, 0.02, 0.3, 0.03])
    ax_complexity = plt.axes([0.1, 0.06, 0.3, 0.03])
    ax_brightness = plt.axes([0.1, 0.10, 0.3, 0.03])
    ax_density = plt.axes([0.1, 0.14, 0.3, 0.03])
    ax_variation = plt.axes([0.1, 0.18, 0.3, 0.03])
    
    # Create sliders
    slider_tempo = Slider(ax_tempo, 'Tempo', 60, 200, valinit=120, valstep=5)
    slider_complexity = Slider(ax_complexity, 'Complexity', 0, 1, valinit=0.5)
    slider_brightness = Slider(ax_brightness, 'Brightness', 0, 1, valinit=0.7)
    slider_density = Slider(ax_density, 'Density', 0, 1, valinit=0.5)
    slider_variation = Slider(ax_variation, 'Variation', 0, 1, valinit=0.3)
    
    # Mode selector
    ax_mode = plt.axes([0.5, 0.02, 0.15, 0.2])
    radio_mode = RadioButtons(
        ax_mode, 
        ('Ionian', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Aeolian', 'Locrian')
    )
    
    # Control buttons
    ax_start = plt.axes([0.7, 0.1, 0.1, 0.04])
    ax_stop = plt.axes([0.85, 0.1, 0.1, 0.04])
    btn_start = Button(ax_start, 'Start')
    btn_stop = Button(ax_stop, 'Stop')
    
    # Update functions
    def update_tempo(val):
        generator.update_parameter('tempo', val)
    
    def update_complexity(val):
        generator.update_parameter('complexity', val)
    
    def update_brightness(val):
        generator.update_parameter('brightness', val)
    
    def update_density(val):
        generator.update_parameter('density', val)
    
    def update_variation(val):
        generator.update_parameter('variation', val)
    
    def update_mode(label):
        generator.update_parameter('mode', label)
    
    def start_generation(event):
        generator.start_generation()
    
    def stop_generation(event):
        generator.stop_generation()
    
    # Connect callbacks
    slider_tempo.on_changed(update_tempo)
    slider_complexity.on_changed(update_complexity)
    slider_brightness.on_changed(update_brightness)
    slider_density.on_changed(update_density)
    slider_variation.on_changed(update_variation)
    radio_mode.on_clicked(update_mode)
    btn_start.on_clicked(start_generation)
    btn_stop.on_clicked(stop_generation)
    
    # Animation function for real-time visualization
    def animate(frame):
        # Get current visualization data
        viz_data = generator.get_visualization_data()
        
        # Update plots
        ax1.clear()
        ax1.set_title('Current Mode Network')
        ax1.text(0.5, 0.5, f"Mode: {viz_data['mode']}", 
                ha='center', va='center', fontsize=20)
        
        # Parameter history
        ax2.clear()
        ax2.set_title('Parameter Evolution')
        for param, history in viz_data['parameter_history'].items():
            if history and param != 'mode':
                times = [h['time'] - history[0]['time'] for h in history]
                values = [h['value'] for h in history]
                ax2.plot(times[-50:], values[-50:], label=param)
        ax2.legend()
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Value')
        
        # Queue status
        ax3.clear()
        ax3.set_title('Generation Queue')
        ax3.bar(['Queue Size'], [viz_data['queue_size']])
        ax3.set_ylim(0, 10)
        
        # Mode relationships
        ax4.clear()
        ax4.set_title('Mode Relationships')
        # Simple visualization of mode graph
        current_mode = viz_data['mode']
        neighbors = viz_data['mode_graph'][current_mode]['neighbors']
        ax4.text(0.5, 0.5, current_mode, ha='center', va='center', 
                fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        angle_step = 2 * np.pi / len(neighbors)
        for i, neighbor in enumerate(neighbors):
            angle = i * angle_step
            x = 0.5 + 0.3 * np.cos(angle)
            y = 0.5 + 0.3 * np.sin(angle)
            ax4.text(x, y, neighbor, ha='center', va='center')
            ax4.plot([0.5, x], [0.5, y], 'k-', alpha=0.3)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        return ax1, ax2, ax3, ax4
    
    # Start animation
    anim = FuncAnimation(fig, animate, interval=500, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return generator


if __name__ == "__main__":
    print("Interactive Modal Music Generator")
    print("=" * 50)
    print("Use the interface to control generation parameters in real-time")
    
    # Create and run interactive interface
    generator = create_interactive_interface() 