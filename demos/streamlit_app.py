"""
SynthNN Interactive Demo App

A Streamlit application showcasing the capabilities of Synthetic Resonant Neural Networks
for music generation, pattern visualization, and real-time interaction.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import wavfile
import io
import base64
from typing import Dict, List, Optional
import time
import sys
import os

# Add project root to Python path to allow synthnn imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Import SynthNN components
from synthnn.core import ResonantNode, ResonantNetwork, SignalProcessor, UniversalPatternCodec
from synthnn.performance import AcceleratedResonantNetwork, BackendManager, BackendType
from synthnn.core.microtonal_extensions import (
    MicrotonalResonantNetwork,
    MicrotonalScaleLibrary,
    AdaptiveMicrotonalSystem,
    MicrotonalScale  # Added MicrotonalScale for type hinting if needed
)
from synthnn.core.emotional_resonance import EmotionalResonanceEngine, EmotionCategory
from synthnn.core import (
    ResonanceField4D, SpatialResonantNode, BoundaryCondition,
    CollectiveIntelligence, CommunicationMode, ConsensusMethod, NetworkRole,
    EvolutionaryResonance, FitnessMetric
)
# Import for Multimodal Demo
from synthnn.core import TextPatternEncoder, ImagePatternEncoder, AudioPatternEncoder # Assuming AudioPatternEncoder exists or will be created
# We might need to define a simplified version or parts of MultiModalResonantSystem here or import it if it becomes a standalone utility

# Page configuration
st.set_page_config(
    page_title="SynthNN Interactive Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .network-vis {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


class SynthNNDemo:
    """Main demo application class."""
    
    def __init__(self):
        self.initialize_session_state()
        self.signal_processor = SignalProcessor()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'network' not in st.session_state:
            st.session_state.network = None
        if 'audio_buffer' not in st.session_state:
            st.session_state.audio_buffer = None
        if 'microtonal_audio_buffer' not in st.session_state:  # New buffer for microtonal audio
            st.session_state.microtonal_audio_buffer = None
        if 'is_playing' not in st.session_state:
            st.session_state.is_playing = False
        if 'backend_type' not in st.session_state:
            st.session_state.backend_type = None
        if 'network_history' not in st.session_state:
            st.session_state.network_history = []
            
    def create_sidebar(self):
        """Create sidebar with controls."""
        with st.sidebar:
            st.header("🎛️ Network Configuration")
            
            # Backend selection
            st.subheader("Compute Backend")
            backend_manager = BackendManager()
            available_backends = backend_manager.list_available_backends()
            
            backend_names = [b.value for b in available_backends]
            selected_backend = st.selectbox(
                "Select Backend",
                backend_names,
                help="Choose compute backend for acceleration"
            )
            
            # Map selection back to BackendType
            backend_map = {b.value: b for b in available_backends}
            st.session_state.backend_type = backend_map[selected_backend]
            
            # Network parameters
            st.subheader("Network Parameters")
            
            num_nodes = st.slider(
                "Number of Nodes",
                min_value=3,
                max_value=50,
                value=10,
                help="Number of resonant nodes in the network"
            )
            
            freq_range = st.slider(
                "Frequency Range (Hz)",
                min_value=50.0,
                max_value=4000.0,
                value=(100.0, 800.0),
                help="Range of frequencies for the nodes"
            )
            
            connection_density = st.slider(
                "Connection Density",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                help="Probability of connection between nodes"
            )
            
            coupling_strength = st.slider(
                "Coupling Strength",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Strength of phase coupling between nodes"
            )
            
            global_damping = st.slider(
                "Global Damping",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Damping factor for amplitude decay"
            )
            
            # Initialize network button
            if st.button("🔄 Initialize Network", type="primary"):
                self.initialize_network(
                    num_nodes, freq_range, connection_density,
                    coupling_strength, global_damping
                )
                st.success("Network initialized!")
                
            # Presets
            st.subheader("Presets")
            preset = st.selectbox(
                "Load Preset",
                ["Custom", "Harmonic Series", "Pentatonic", "Chaotic", "Bell-like"]
            )
            
            if preset != "Custom" and st.button("Load Preset"):
                self.load_preset(preset)
                
            return {
                'num_nodes': num_nodes,
                'freq_range': freq_range,
                'connection_density': connection_density,
                'coupling_strength': coupling_strength,
                'global_damping': global_damping
            }
            
    def initialize_network(self, num_nodes, freq_range, connection_density,
                          coupling_strength, global_damping):
        """Initialize the resonant network with given parameters."""
        # Create accelerated network
        network = AcceleratedResonantNetwork(
            name="demo_network",
            backend=st.session_state.backend_type
        )
        
        # Set network parameters
        network.coupling_strength = coupling_strength
        network.global_damping = global_damping
        
        # Add nodes with frequencies in specified range
        frequencies = np.linspace(freq_range[0], freq_range[1], num_nodes)
        for i, freq in enumerate(frequencies):
            # Add some randomness
            freq_with_noise = freq + np.random.uniform(-10, 10)
            node = ResonantNode(
                node_id=f"node_{i}",
                frequency=freq_with_noise,
                phase=np.random.uniform(0, 2*np.pi),
                amplitude=1.0 / np.sqrt(num_nodes)
            )
            network.add_node(node)
            
        # Create connections based on density
        node_ids = list(network.nodes.keys())
        for i, src in enumerate(node_ids):
            for j, tgt in enumerate(node_ids):
                if i != j and np.random.random() < connection_density:
                    weight = np.random.uniform(-0.5, 0.5)
                    network.connect(src, tgt, weight)
                    
        st.session_state.network = network
        st.session_state.network_history = []
        
    def load_preset(self, preset_name):
        """Load a preset network configuration."""
        presets = {
            "Harmonic Series": {
                'frequencies': [110 * i for i in range(1, 11)],  # A2 harmonics
                'connection_pattern': 'cascade',
                'coupling': 1.0
            },
            "Pentatonic": {
                'frequencies': [220, 247, 294, 330, 370, 440, 494, 587, 659, 740],  # A pentatonic
                'connection_pattern': 'circular',
                'coupling': 3.0
            },
            "Chaotic": {
                'frequencies': np.random.uniform(100, 1000, 15),
                'connection_pattern': 'random',
                'coupling': 5.0
            },
            "Bell-like": {
                'frequencies': [200, 373, 447, 587, 719, 897, 1123],  # Bell partials
                'connection_pattern': 'full',
                'coupling': 0.5
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            network = AcceleratedResonantNetwork(
                name=f"{preset_name}_network",
                backend=st.session_state.backend_type
            )
            
            # Add nodes
            for i, freq in enumerate(preset['frequencies']):
                node = ResonantNode(
                    node_id=f"node_{i}",
                    frequency=freq,
                    phase=np.random.uniform(0, 2*np.pi),
                    amplitude=1.0 / np.sqrt(len(preset['frequencies']))
                )
                network.add_node(node)
                
            # Create connections based on pattern
            node_ids = list(network.nodes.keys())
            if preset['connection_pattern'] == 'cascade':
                for i in range(len(node_ids) - 1):
                    network.connect(node_ids[i], node_ids[i+1], 0.5)
            elif preset['connection_pattern'] == 'circular':
                for i in range(len(node_ids)):
                    network.connect(node_ids[i], node_ids[(i+1) % len(node_ids)], 0.3)
            elif preset['connection_pattern'] == 'random':
                for i in range(len(node_ids)):
                    for j in range(len(node_ids)):
                        if i != j and np.random.random() < 0.3:
                            network.connect(node_ids[i], node_ids[j], 
                                          np.random.uniform(-0.5, 0.5))
            elif preset['connection_pattern'] == 'full':
                for i in range(len(node_ids)):
                    for j in range(len(node_ids)):
                        if i != j:
                            network.connect(node_ids[i], node_ids[j], 0.1)
                            
            network.coupling_strength = preset['coupling']
            st.session_state.network = network
            st.session_state.network_history = []
            
    def visualize_network(self, network):
        """Create network visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Network structure visualization
        G = nx.DiGraph()
        for node_id in network.nodes:
            G.add_node(node_id)
            
        for (src, tgt), conn in network.connections.items():
            G.add_edge(src, tgt, weight=conn.weight)
            
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes with color based on amplitude
        node_colors = [network.nodes[n].amplitude for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              cmap='viridis', node_size=500, ax=ax1)
        
        # Draw edges with width based on weight
        edges = G.edges()
        weights = [abs(G[u][v]['weight']) * 3 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax1)
        
        # Labels
        nx.draw_networkx_labels(G, pos, ax=ax1)
        ax1.set_title("Network Structure")
        ax1.axis('off')
        
        # Phase visualization
        phases = [network.nodes[n].phase for n in network.nodes]
        frequencies = [network.nodes[n].frequency for n in network.nodes]
        amplitudes = [network.nodes[n].amplitude for n in network.nodes]
        
        # Polar plot of phases
        ax2 = plt.subplot(122, projection='polar')
        for phase, freq, amp in zip(phases, frequencies, amplitudes):
            ax2.plot([0, phase], [0, amp], 'o-', markersize=8, 
                    linewidth=2, alpha=0.7)
            
        ax2.set_ylim(0, max(amplitudes) * 1.2 if amplitudes else 1)
        ax2.set_title("Phase Distribution")
        
        plt.tight_layout()
        return fig
        
    def generate_audio(self, network, duration=2.0, sample_rate=44100):
        """Generate audio from network."""
        audio = network.generate_signals(duration, sample_rate)
        
        # Apply envelope for smooth start/end
        envelope = np.ones_like(audio)
        fade_samples = int(0.05 * sample_rate)  # 50ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= envelope
        
        return audio
        
    def analyze_audio(self, audio, sample_rate=44100):
        """Analyze generated audio."""
        # Set signal processor sample rate
        self.signal_processor.sample_rate = sample_rate
        
        # Compute spectrum
        spectrum_data = self.signal_processor.analyze_spectrum(audio)
        spectrum = {
            'frequencies': spectrum_data[0],
            'magnitudes': spectrum_data[1]
        }
        
        # Find fundamental frequency
        fundamental = self.signal_processor.extract_fundamental(audio, freq_range=(50, 2000))
        
        # Compute phase coherence over time
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = window_size // 2
        
        coherence_values = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            coherence = self.signal_processor.compute_phase_coherence(
                {"signal": window}
            )
            coherence_values.append(coherence[0, 0])
            
        return spectrum, fundamental, coherence_values
        
    def create_audio_player(self, audio, sample_rate=44100):
        """Create an audio player widget."""
        # Convert to WAV format
        buffer = io.BytesIO()
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        
        # Encode as base64
        audio_bytes = buffer.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Create HTML audio element
        audio_html = f"""
        <audio controls>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        
        return audio_html
        
    def run_simulation_step(self, network, dt=0.01, external_input=None):
        """Run one simulation step and record state."""
        # Apply external input if provided
        if external_input:
            network.step(dt, external_input)
        else:
            network.step(dt)
            
        # Record state
        state = {
            'time': network.time,
            'phases': [network.nodes[n].phase for n in network.nodes],
            'amplitudes': [network.nodes[n].amplitude for n in network.nodes],
            'synchronization': network.measure_synchronization()
        }
        
        return state
        
    def main(self):
        """Main application logic."""
        st.title("🎵 SynthNN Interactive Demo")
        st.markdown("""
        Explore the fascinating world of **Synthetic Resonant Neural Networks** - 
        a novel approach combining wave physics, resonance theory, and neural computation
        for music generation and pattern recognition.
        """)
        
        # Create sidebar
        params = self.create_sidebar()
        
        # Main content area
        if st.session_state.network is None:
            st.info("👈 Please initialize a network using the sidebar controls")
            return
            
        network = st.session_state.network
        
        # Create tabs
        tab_music_gen, tab_viz, tab_analysis, tab_interactive, tab_microtonal, tab_multimodal, tab_emotional, tab_4d_field, tab_collective, tab_evolution = st.tabs([
            "🎼 Music Generation", 
            "📊 Network Visualization", 
            "🔬 Analysis",
            "🎮 Interactive Control",
            "🌍 Microtonal Exploration",
            "👁️‍🗨️ Multimodal Playground",
            "💖 Emotional Resonance",
            "🌊 4D Resonance Fields",  # New Tab
            "🧠 Collective Intelligence",  # New Tab
            "🧬 Evolutionary Resonance"  # New Tab
        ])

        with tab_music_gen:
            st.header("Music Generation")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                duration = st.slider("Duration (seconds)", 0.5, 5.0, 2.0)
                
            with col2:
                if st.button("🎵 Generate Audio"):
                    with st.spinner("Generating audio..."):
                        audio = self.generate_audio(network, duration)
                        st.session_state.audio_buffer = audio
                        
            with col3:
                if st.session_state.audio_buffer is not None:
                    # Download button
                    buffer = io.BytesIO()
                    audio_int16 = (st.session_state.audio_buffer * 32767).astype(np.int16)
                    wavfile.write(buffer, 44100, audio_int16)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Download",
                        data=buffer,
                        file_name="synthnn_output.wav",
                        mime="audio/wav"
                    )
                    
            # Audio player
            if st.session_state.audio_buffer is not None:
                st.markdown("### Generated Audio")
                audio_html = self.create_audio_player(st.session_state.audio_buffer)
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Waveform visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                time_axis = np.arange(len(st.session_state.audio_buffer)) / 44100
                ax.plot(time_axis, st.session_state.audio_buffer, linewidth=0.5)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Waveform")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
        with tab_viz:
            st.header("Network Visualization")
            
            # Real-time simulation controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sim_steps = st.slider("Simulation Steps", 10, 200, 50)
                
            with col2:
                if st.button("▶️ Run Simulation"):
                    progress_bar = st.progress(0)
                    
                    for i in range(sim_steps):
                        state = self.run_simulation_step(network)
                        st.session_state.network_history.append(state)
                        progress_bar.progress((i + 1) / sim_steps)
                        
                    st.success("Simulation complete!")
                    
            with col3:
                if st.button("🔄 Reset"):
                    st.session_state.network_history = []
                    
            # Network visualization
            with st.container():
                fig = self.visualize_network(network)
                st.pyplot(fig)
                
            # Plot synchronization over time
            if st.session_state.network_history:
                sync_values = [s['synchronization'] for s in st.session_state.network_history]
                
                fig_sync, ax_sync = plt.subplots(figsize=(10, 4))
                ax_sync.plot(sync_values, marker='.')
                ax_sync.set_title("Network Synchronization Over Time")
                ax_sync.set_xlabel("Time Step")
                ax_sync.set_ylabel("Synchronization Index")
                ax_sync.grid(True, alpha=0.3)
                st.pyplot(fig_sync)
                
        with tab_analysis:
            st.header("Analysis of Generated Audio")
            
            if st.session_state.audio_buffer is None:
                st.warning("Please generate audio first in the 'Music Generation' tab.")
            else:
                audio = st.session_state.audio_buffer
                spectrum, fundamental, coherence = self.analyze_audio(audio)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fundamental Frequency", f"{fundamental:.2f} Hz" if fundamental else "N/A")
                with col2:
                    st.metric("Average Phase Coherence", f"{np.mean(coherence):.3f}" if coherence else "N/A")
                
                # Spectrum plot
                fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
                ax_spec.plot(spectrum['frequencies'], spectrum['magnitudes'])
                ax_spec.set_title("Frequency Spectrum")
                ax_spec.set_xlabel("Frequency (Hz)")
                ax_spec.set_ylabel("Magnitude")
                ax_spec.set_xlim(0, 2000) # Limit to 2kHz for clarity
                ax_spec.grid(True, alpha=0.3)
                st.pyplot(fig_spec)
                
                # Coherence plot
                fig_coh, ax_coh = plt.subplots(figsize=(10, 4))
                ax_coh.plot(coherence)
                ax_coh.set_title("Phase Coherence Over Time")
                ax_coh.set_xlabel("Time Window")
                ax_coh.set_ylabel("Coherence")
                ax_coh.grid(True, alpha=0.3)
                st.pyplot(fig_coh)
                
        with tab_interactive:
            st.header("Interactive Network Control")
            
            st.markdown("""
            **Coming Soon!** This section will allow real-time manipulation of network parameters 
            and external inputs to observe emergent behaviors.
            """)
            
            # Placeholder for interactive elements
            # Example: slider for external input to a node
            if len(network.nodes) > 0:
                selected_node = st.selectbox(
                    "Select Node to Excite", 
                    list(network.nodes.keys()),
                    key="interactive_node_select"
                )
                excitation_strength = st.slider(
                    "Excitation Strength", 0.0, 1.0, 0.1, 
                    key="interactive_excitation"
                )
                
                if st.button("⚡ Excite Node"):
                    external_input = {selected_node: excitation_strength}
                    state = self.run_simulation_step(network, external_input=external_input)
                    st.session_state.network_history.append(state)
                    st.success(f"Applied excitation to {selected_node}")

        with tab_microtonal: # Content for the new Microtonal Exploration tab
            st.header("🌍 Microtonal Exploration")
            st.markdown("""
            Explore the rich world of microtonal music with SynthNN. 
            This section demonstrates the network's ability to handle diverse tuning systems, 
            continuous pitch manipulations, and adaptive learning of scales.
            """)

            # Sub-sections for different microtonal demos
            microtonal_demo_option = st.selectbox(
                "Select Microtonal Demonstration",
                [
                    "Microtonal Scale Showcase",
                    "Continuous Pitch Field",
                    "Maqam Modulation",
                    "Adaptive Scale Learning",
                    "Comma Pump Effect"
                ]
            )

            if microtonal_demo_option == "Microtonal Scale Showcase":
                self.render_microtonal_scale_showcase()
            elif microtonal_demo_option == "Continuous Pitch Field":
                self.render_continuous_pitch_field_demo()
            elif microtonal_demo_option == "Maqam Modulation":
                self.render_maqam_modulation_demo()
            elif microtonal_demo_option == "Adaptive Scale Learning":
                self.render_adaptive_learning_demo()
            elif microtonal_demo_option == "Comma Pump Effect":
                self.render_comma_pump_demo()

        with tab_multimodal:
            self.render_multimodal_playground_demo()

        with tab_emotional:
            st.header("💖 Emotional Resonance Engine")
            st.markdown("""
            Experience how emotions map to resonant frequencies and harmonic patterns.
            This engine can analyze emotional content, generate empathetic responses,
            and create emotional journeys through sound.
            """)
            
            self.render_emotional_resonance_demo()

        # New Advanced Features Tabs
        with tab_4d_field:
            resonance_field_4d_demo()

        with tab_collective:
            collective_intelligence_demo()

        with tab_evolution:
            evolutionary_resonance_demo()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Built with ❤️ using SynthNN - Synthetic Resonant Neural Networks</p>
            <p>Explore the intersection of neuroscience, physics, and music</p>
        </div>
        """, unsafe_allow_html=True)

    def render_microtonal_scale_showcase(self):
        st.subheader("🎵 Microtonal Scale Showcase")
        st.markdown("Visualizing various microtonal scales from different cultures and theories.")

        scales = MicrotonalScaleLibrary.get_scales()
        
        # Allow user to select multiple scales for comparison or a single scale for audio demo
        selected_scale_names = st.multiselect(
            "Select scales to visualize:",
            list(scales.keys()),
            default=['just_major', 'maqam_rast', 'slendro']
        )

        if not selected_scale_names:
            st.warning("Please select at least one scale.")
            return

        num_selected = len(selected_scale_names)
        if num_selected > 0:
            cols = st.columns(min(num_selected, 3)) # Max 3 plots per row
            for i, scale_name in enumerate(selected_scale_names):
                scale = scales[scale_name]
                with cols[i % 3]:
                    st.markdown(f"**{scale.name}**")
                    st.caption(scale.description)
                    cents = scale.to_cents()
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.scatter(range(len(cents[:12])), cents[:12], s=30, color='#1f77b4')
                    ax.plot(range(len(cents[:12])), cents[:12], alpha=0.6, color='#1f77b4')
                    ax.set_xlabel('Scale Degree')
                    ax.set_ylabel('Cents')
                    ax.set_title(f"{scale.name} Structure", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

        st.divider()
        st.markdown("### Audio Demonstration of a Selected Scale")
        audio_demo_scale_name = st.selectbox(
            "Select a scale for audio demonstration:",
            list(scales.keys()),
            index=list(scales.keys()).index('maqam_rast') if 'maqam_rast' in scales else 0
        )
        
        if st.button(f"Generate Audio for {audio_demo_scale_name}"):
            scale = scales[audio_demo_scale_name]
            network = MicrotonalResonantNetwork(
                name=f"{scale.name}_audio_demo",
                scale=scale,
                base_freq=261.63 # C4
            )
            network.create_scale_nodes(num_octaves=1)
            network.create_modal_connections('harmonic_series', weight_scale=0.2)
            
            with st.spinner(f"Generating audio for {scale.name}..."):
                # Simple arpeggio-like pattern
                duration_per_note = 0.5
                total_duration = len(scale.intervals) * duration_per_note
                sample_rate = 22050 # Lower sample rate for faster generation in demo
                
                full_audio = np.array([])
                for i in range(len(scale.intervals)):
                    # Target the node corresponding to the current scale degree
                    node_id_to_excite = f"degree_{i}"
                    original_amplitudes = {nid: n.amplitude for nid, n in network.nodes.items()}

                    for nid, node_obj in network.nodes.items(): # Corrected variable name
                        if nid == node_id_to_excite:
                            node_obj.amplitude = 1.0 # Excite the target node
                        else:
                            node_obj.amplitude = 0.1 # Dampen others

                    segment = network.generate_signals(duration_per_note, sample_rate)
                    full_audio = np.concatenate((full_audio, segment))
                    
                    # Reset amplitudes for next note or use a decay mechanism
                    for nid, node_obj in network.nodes.items(): # Corrected variable name
                         node_obj.amplitude = original_amplitudes[nid] * 0.5 # Basic decay
                
                st.session_state.microtonal_audio_buffer = full_audio

        if st.session_state.microtonal_audio_buffer is not None:
            st.audio(st.session_state.microtonal_audio_buffer, sample_rate=22050)
            st.session_state.microtonal_audio_buffer = None # Clear after playing

    def render_continuous_pitch_field_demo(self):
        st.subheader("🌊 Continuous Pitch Field")
        st.markdown("Demonstrating networks with continuously distributed pitches and smooth glissandi. Nodes are distributed using a chosen mathematical series, and connections are formed based on spectral relationships.")

        col1, col2 = st.columns(2)
        with col1:
            num_nodes = st.slider("Number of Nodes in Field", 5, 30, 15, key="cpf_nodes")
            distribution = st.selectbox(
                "Node Distribution Type",
                ['logarithmic', 'linear', 'golden'],
                index=2, # Default to golden
                key="cpf_dist"
            )
        with col2:
            freq_min = st.slider("Min Frequency (Hz)", 50.0, 500.0, 220.0, key="cpf_min_freq")
            freq_max = st.slider("Max Frequency (Hz)", 501.0, 2000.0, 880.0, key="cpf_max_freq")
        
        harmonicity_threshold = st.slider(
            "Harmonicity Threshold for Connections", 
            0.05, 0.5, 0.15, 
            help="Lower values create more connections based on looser harmonic relationships.",
            key="cpf_harm_thresh"
        )

        if st.button("Initialize & Generate Continuous Pitch Field Audio", key="cpf_generate"):
            network = MicrotonalResonantNetwork(name="continuous_field_demo")
            network.create_continuous_pitch_field(
                freq_range=(freq_min, freq_max),
                num_nodes=num_nodes,
                distribution=distribution
            )
            network.create_spectral_connections(harmonicity_threshold=harmonicity_threshold)
            st.session_state.cpf_network = network # Store for visualization

            st.write(f"Created {len(network.nodes)} nodes ({distribution} distribution) and connected them based on spectral relationships.")

            # Set some glissando targets for demonstration
            node_ids = list(network.nodes.keys())
            if len(node_ids) > 0:
                for i in range(0, len(node_ids), 3):
                    current_freq = network.nodes[node_ids[i]].frequency
                    # Randomly glide up or down by a small factor
                    target_freq = current_freq * np.random.uniform(0.8, 1.25) 
                    network.glissando_to_pitch(node_ids[i], target_freq)
                for i in range(1, len(node_ids), 4):
                    current_freq = network.nodes[node_ids[i]].frequency
                    target_freq = current_freq * np.random.uniform(1.2, 1.6)
                    network.glissando_to_pitch(node_ids[i], target_freq)

            with st.spinner("Generating evolving microtonal texture..."):
                audio = network.generate_microtonal_texture(
                    duration=st.session_state.get('cpf_duration', 5.0),
                    density=st.session_state.get('cpf_density', 0.7),
                    evolution_rate=st.session_state.get('cpf_evo_rate', 0.25),
                    sample_rate=22050
                )
                st.session_state.microtonal_audio_buffer = audio
                st.session_state.cpf_pitch_trajectories = network.pitch_trajectories
        
        st.slider("Audio Duration (s)", 2.0, 10.0, 5.0, key="cpf_duration")
        st.slider("Node Activation Density", 0.1, 1.0, 0.7, key="cpf_density")
        st.slider("Timbral Evolution Rate", 0.05, 0.5, 0.25, key="cpf_evo_rate")

        if st.session_state.get('microtonal_audio_buffer') is not None:
            st.audio(st.session_state.microtonal_audio_buffer, sample_rate=22050)
            
            if st.session_state.get('cpf_pitch_trajectories'):
                fig, ax = plt.subplots(figsize=(10, 4))
                for node_id, trajectory in st.session_state.cpf_pitch_trajectories.items():
                    if len(trajectory) > 1:
                        time_axis = np.linspace(0, st.session_state.cpf_duration, len(trajectory))
                        ax.plot(time_axis, trajectory, alpha=0.6, linewidth=1.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title('Pitch Trajectories of Nodes in Continuous Field')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.session_state.microtonal_audio_buffer = None # Clear after use
            st.session_state.cpf_pitch_trajectories = None

    def render_maqam_modulation_demo(self):
        st.subheader("🔄 Maqam Modulation")
        st.markdown("Showcasing smooth transitions between different Arabic Maqamat (e.g., Rast to Bayati). This demonstrates the network's ability to dynamically retune and glide between complex tuning systems.")
        
        scales = MicrotonalScaleLibrary.get_scales()
        maqam_scales = {k: v for k, v in scales.items() if 'maqam' in k.lower() or k in ['just_major', 'pythagorean']}

        col1, col2 = st.columns(2)
        with col1:
            start_maqam_name = st.selectbox("Starting Maqam/Scale", list(maqam_scales.keys()), index=list(maqam_scales.keys()).index('maqam_rast'), key="mm_start")
        with col2:
            end_maqam_name = st.selectbox("Target Maqam/Scale", list(maqam_scales.keys()), index=list(maqam_scales.keys()).index('maqam_bayati'), key="mm_end")

        base_freq_options = {"C4 (261.63 Hz)": 261.63, "D4 (293.66 Hz)": 293.66, "A3 (220 Hz)": 220.0}
        selected_base_freq_name = st.selectbox("Base Frequency for Tonic", list(base_freq_options.keys()), key="mm_base_freq")
        base_freq = base_freq_options[selected_base_freq_name]

        modulation_time = st.slider("Modulation Duration (s)", 1.0, 5.0, 2.0, key="mm_mod_time")
        phrase_time = st.slider("Phrase Duration Before/After Mod (s)", 1.0, 4.0, 2.0, key="mm_phrase_time")

        if st.button("Generate Maqam Modulation Audio", key="mm_generate"):
            start_scale = maqam_scales[start_maqam_name]
            end_scale = maqam_scales[end_maqam_name]

            network = MicrotonalResonantNetwork(
                name="maqam_modulation_demo",
                base_freq=base_freq,
                scale=start_scale
            )
            network.create_scale_nodes(num_octaves=1) # Keep it simple for demo
            network.create_modal_connections('harmonic_series', weight_scale=0.15)

            sample_rate = 22050
            audio_segments = []

            with st.spinner(f"Generating initial phrase in {start_maqam_name}..."):
                # Generate audio in starting Maqam
                # Excite all nodes briefly to establish the mode
                for node in network.nodes.values(): node.amplitude = 0.8
                initial_phrase = network.generate_signals(phrase_time, sample_rate, excite_all_nodes_momentarily=True)
                audio_segments.append(initial_phrase)
            
            with st.spinner(f"Modulating from {start_maqam_name} to {end_maqam_name}..."):
                network.scale = end_scale # Switch to the target scale definition
                # Retune nodes to the new maqam using glissando
                max_nodes_to_retune = len(network.nodes)
                for i, node_id in enumerate(list(network.nodes.keys())[:max_nodes_to_retune]):
                    # Calculate target frequency in the new scale, possibly shifting octave for best fit
                    # This is a simplified approach; more sophisticated voice leading could be used.
                    degree_in_new_scale = i % len(end_scale.intervals)
                    new_freq = end_scale.get_frequency(network.base_freq, degree_in_new_scale)
                    network.glissando_to_pitch(node_id, new_freq)
                
                modulation_audio_list = [] # Changed to list
                num_mod_samples = int(modulation_time * sample_rate)
                for _ in range(num_mod_samples):
                    network.step_with_glissando(1.0 / sample_rate)
                    signals = network.get_signals()
                    modulation_audio_list.append(sum(signals.values()) * 0.7) # Apply some scaling
                audio_segments.append(np.array(modulation_audio_list))

            with st.spinner(f"Generating final phrase in {end_maqam_name}..."):
                # Generate audio in ending Maqam
                for node in network.nodes.values(): node.amplitude = 0.8 
                final_phrase = network.generate_signals(phrase_time, sample_rate, excite_all_nodes_momentarily=True)
                audio_segments.append(final_phrase)

            full_audio = np.concatenate(audio_segments)
            full_audio = full_audio / np.max(np.abs(full_audio)) if np.max(np.abs(full_audio)) > 0 else full_audio
            st.session_state.microtonal_audio_buffer = full_audio
            st.success("Maqam modulation audio generated!")

        if st.session_state.get('microtonal_audio_buffer') is not None:
            st.audio(st.session_state.microtonal_audio_buffer, sample_rate=22050)
            st.session_state.microtonal_audio_buffer = None # Clear after playing

    def render_adaptive_learning_demo(self):
        st.subheader("🧠 Adaptive Scale Learning")
        st.markdown("Demonstrating the network's ability to learn scales from audio examples (simulated here) and find consonant intervals using its resonant properties.")

        st.markdown("**1. Simulate a Performance & Learn Scale**")
        # Create a simple melodic pattern to simulate a performance
        sample_rate = 22050 # Keep consistent for demo
        duration = 3.0
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        
        # Melody based on a known microtonal scale for clear demonstration
        # Using a subset of Maqam Rast intervals for this example
        # Tonic (1.0), Minor Third (approx 6/5 or 315 cents), Perfect Fifth (3/2)
        # Frequencies for C4 tonic (261.63 Hz)
        # C4: 261.63 Hz
        # E-flat (approx): 261.63 * (6/5) = 313.95 Hz
        # G4: 261.63 * (3/2) = 392.44 Hz
        # C5: 261.63 * 2 = 523.26 Hz
        sim_frequencies = [261.63, 313.95, 392.44, 523.26, 392.44, 313.95, 261.63]
        melody = np.array([])
        
        segment_duration = duration / len(sim_frequencies)
        segment_samples = int(segment_duration * sample_rate)
        time_per_segment = np.linspace(0, segment_duration, segment_samples, endpoint=False)

        for freq in sim_frequencies:
            segment = np.sin(2 * np.pi * freq * time_per_segment) * np.exp(-time_per_segment * 2) # Add decay
            melody = np.concatenate((melody, segment))
        
        st.markdown("Simulated performance audio (short excerpt):")
        st.audio(melody, sample_rate=sample_rate)

        if st.button("Learn Scale from Simulated Performance", key="al_learn"):
            base_network = MicrotonalResonantNetwork("adaptive_learning_network")
            adaptive_system = AdaptiveMicrotonalSystem(base_network)
            
            with st.spinner("Analyzing audio and learning scale..."):
                learned_scale = adaptive_system.learn_scale_from_performance(
                    melody, sample_rate, "learned_from_sim_perf"
                )
            
            st.session_state.al_learned_scale = learned_scale
            st.success(f"Successfully learned scale: {learned_scale.name}")

        if st.session_state.get('al_learned_scale') is not None:
            learned_scale = st.session_state.al_learned_scale
            st.markdown(f"**Learned Scale: {learned_scale.name}**")
            st.write(f"Description: {learned_scale.description}")
            st.write(f"Intervals (ratios): {[f'{r:.4f}' for r in learned_scale.intervals]}")
            st.write(f"Intervals (cents): {[f'{c:.2f}' for c in learned_scale.to_cents()]}")
            
            fig, ax = plt.subplots(figsize=(6,3))
            cents = learned_scale.to_cents()
            ax.scatter(range(len(cents)), cents, s=40)
            ax.plot(range(len(cents)), cents, alpha=0.5)
            ax.set_title("Structure of Learned Scale")
            ax.set_xlabel("Scale Degree Index")
            ax.set_ylabel("Cents from Tonic")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.divider()
        st.markdown("**2. Find Consonant Intervals using Network Resonance**")
        res_min_ratio = st.slider("Min Ratio for Consonance Search", 1.0, 1.9, 1.4, key="al_min_ratio") # Default: perfect fourth
        res_max_ratio = st.slider("Max Ratio for Consonance Search", float(res_min_ratio) + 0.1, 2.5, 2.0, key="al_max_ratio") # Default: octave
        res_cents = st.slider("Resolution for Search (cents)", 1.0, 20.0, 5.0, key="al_res_cents")

        if st.button("Find Consonant Intervals via Resonance", key="al_find_consonance"):
            if 'al_learned_scale' not in st.session_state or st.session_state.al_learned_scale is None:
                 # Use a default network if no scale has been learned yet for this part of the demo
                 temp_scale = MicrotonalScaleLibrary.get_scales()['just_major']
                 base_network = MicrotonalResonantNetwork("consonance_finder_network", scale=temp_scale)
            else:
                 base_network = MicrotonalResonantNetwork("consonance_finder_network", scale=st.session_state.al_learned_scale)
            adaptive_system = AdaptiveMicrotonalSystem(base_network)
            
            with st.spinner("Searching for consonant intervals using network resonance..."):
                consonant_ratios = adaptive_system.find_consonant_intervals(
                    freq_range=(res_min_ratio, res_max_ratio),
                    resolution_cents=res_cents
                )
            st.session_state.al_consonant_ratios = consonant_ratios
            st.success(f"Found {len(consonant_ratios)} potentially consonant intervals.")

        if st.session_state.get('al_consonant_ratios') is not None:
            consonant_ratios = st.session_state.al_consonant_ratios
            if consonant_ratios:
                st.markdown("**Discovered Consonant Intervals (by phase-locking):**")
                data = []
                for ratio in consonant_ratios:
                    cents = 1200 * np.log2(ratio)
                    data.append({"Ratio": f"{ratio:.4f}", "Cents": f"{cents:.2f}"})
                st.table(data)
            else:
                st.info("No strong consonant intervals found with current settings. Try adjusting search range or resolution.")

    def render_comma_pump_demo(self):
        st.subheader("🔬 Comma Pump Effect")
        st.markdown("Exploring microtonal shifts using comma pumps. A comma is a small interval resulting from discrepancies in tuning systems (e.g., the difference between four perfect fifths and two octaves plus a major third, which is the syntonic comma).")
        
        scales = MicrotonalScaleLibrary.get_scales()
        # Select scales that are typically used with comma adjustments or show them clearly
        relevant_scales = {k:v for k,v in scales.items() if k in ['just_major', 'pythagorean', 'maqam_rast']}
        if not relevant_scales: relevant_scales = {list(scales.keys())[0]: list(scales.values())[0]} # Fallback
        
        selected_scale_name = st.selectbox("Base Scale for Comma Pump Demo", list(relevant_scales.keys()), key="cp_scale")
        initial_scale = relevant_scales[selected_scale_name]

        comma_types = {
            'Syntonic Comma (81/80)': 'syntonic',
            'Pythagorean Comma (531441/524288)': 'pythagorean',
            'Diaschisma (2048/2025)': 'diaschisma',
            'Schisma (32805/32768)': 'schisma'
        }
        selected_comma_name = st.selectbox("Select Comma Type to Apply", list(comma_types.keys()), key="cp_comma_type")
        comma_type_key = comma_types[selected_comma_name]

        num_pumps = st.slider("Number of Comma Pumps to Apply", 1, 5, 1, key="cp_num_pumps")
        base_freq = st.number_input("Base Frequency (Hz)", value=261.63, key="cp_base_freq") # C4
        
        if st.button("Generate Audio with Comma Pump", key="cp_generate"):
            network = MicrotonalResonantNetwork(
                name="comma_pump_network",
                scale=initial_scale,
                base_freq=base_freq
            )
            network.create_scale_nodes(num_octaves=1)
            network.create_modal_connections('full', weight_scale=0.1)

            sample_rate = 22050
            duration_per_segment = 1.5 # Seconds per segment (before/after pump)
            audio_segments = []
            pitch_history_per_node = {node_id: [node.frequency] for node_id, node in network.nodes.items()}

            # 1. Audio with original tuning
            with st.spinner("Generating audio with original tuning..."):
                # Excite nodes to establish tonality
                for node in network.nodes.values(): node.amplitude = 0.7
                segment_original = network.generate_signals(duration_per_segment, sample_rate, excite_all_nodes_momentarily=True)
                audio_segments.append(segment_original)
            
            # Record frequencies after initial segment generation
            for node_id, node in network.nodes.items():
                 pitch_history_per_node[node_id].append(node.frequency)

            # 2. Apply comma pump(s) and generate audio during/after glissando
            with st.spinner(f"Applying {num_pumps}x {selected_comma_name} and generating audio..."):
                for _ in range(num_pumps):
                    network.apply_comma_pump(comma_type_key)
                
                # Audio during glissando settle
                num_gliss_settle_samples = int(duration_per_segment * sample_rate)
                gliss_audio_list = []
                for k_sample in range(num_gliss_settle_samples):
                    network.step_with_glissando(1.0 / sample_rate)
                    signals = network.get_signals()
                    gliss_audio_list.append(sum(signals.values()) * 0.6)
                    if k_sample % (sample_rate // 20) == 0: # Record pitch history 20x per second
                        for node_id, node in network.nodes.items():
                            pitch_history_per_node[node_id].append(node.frequency)
                
                audio_segments.append(np.array(gliss_audio_list))
            
            # Record final frequencies
            for node_id, node in network.nodes.items():
                 pitch_history_per_node[node_id].append(node.frequency)

            full_audio = np.concatenate(audio_segments)
            full_audio = full_audio / np.max(np.abs(full_audio)) if np.max(np.abs(full_audio)) > 0 else full_audio
            st.session_state.microtonal_audio_buffer = full_audio
            st.session_state.cp_pitch_history = pitch_history_per_node
            st.success("Comma pump audio generated!")

        if st.session_state.get('microtonal_audio_buffer') is not None:
            st.audio(st.session_state.microtonal_audio_buffer, sample_rate=22050)
            
            if st.session_state.get('cp_pitch_history'):
                fig, ax = plt.subplots(figsize=(10, 4))
                history = st.session_state.cp_pitch_history
                # Plot trajectories for a few representative nodes
                num_nodes_to_plot = min(5, len(history.keys()))
                node_ids_to_plot = list(history.keys())[:num_nodes_to_plot]
                
                for node_id in node_ids_to_plot:
                    trajectory = history[node_id]
                    # Ensure x-axis matches number of points recorded
                    time_axis = np.arange(len(trajectory))
                    ax.plot(time_axis, trajectory, alpha=0.7, linewidth=1.5, label=f"Node {node_id.split('_')[-1]}")
                
                ax.set_xlabel('Time Steps (arbitrary units, captures pre/during/post pump)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(f"Pitch Trajectories During {selected_comma_name} Pump")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.session_state.microtonal_audio_buffer = None
            st.session_state.cp_pitch_history = None

    def render_multimodal_playground_demo(self):
        st.subheader("👁️‍🗨️ Multimodal Playground")
        st.markdown("Translate between text, images, and sound patterns using resonant networks. "
                    "This demonstrates SynthNN's potential for cross-modal understanding and generation.")

        if 'mm_language_network' not in st.session_state or st.session_state.mm_language_network is None:
            st.session_state.mm_language_network = self._create_simplified_language_network_for_demo()
            # This ensures linguistic_feature_bands_for_mm_demo is set on self before being used below
        
        # Make linguistic_features accessible in the activation logic
        # This should ideally be part of the network or a shared config
        # For now, accessing it via self if the network creation method sets it on self.
        linguistic_features = getattr(self, 'linguistic_feature_bands_for_mm_demo', { # Fallback if not set
            'chars': np.linspace(100, 300, 10),    
            'words': np.linspace(300, 700, 15),    
            'semantic_concepts': np.linspace(700, 1200, 10) 
        })

        lang_network = st.session_state.mm_language_network
        # text_encoder = TextPatternEncoder() # We are not using its direct output for activation keys anymore

        input_modality = st.selectbox(
            "Input Modality",
            ["Text", "Image (coming soon)", "Audio (coming soon)"],
            key="mm_input_modality"
        )

        output_modalities = st.multiselect(
            "Desired Output Modalities",
            ["Audio Pattern", "Visual Pattern (Network State)", "Textual Description (Interpretation)"],
            default=["Audio Pattern"],
            key="mm_output_modalities"
        )

        if input_modality == "Text":
            text_input = st.text_area("Enter Text:", "Resonance in neural networks creates music.", key="mm_text_input")
            if st.button("Process Text and Generate Outputs", key="mm_process_text"):
                if not text_input.strip():
                    st.warning("Please enter some text.")
                    return
                
                with st.spinner("Processing text and generating multimodal outputs..."):
                    st.write(f"**Input Text:** {text_input}")
                    
                    # Reset network or selectively update nodes
                    # active_node_ids_from_encoding = set() # We are not relying on encoder\'s keys anymore for direct activation

                    # --- New Activation Logic based on pre-defined network nodes ---
                    st.write("**Applying new activation logic...**")
                    activated_in_lang_network = 0
                    text_lower = text_input.lower()
                    words_in_text = set(text_lower.split())

                    for node_id, node_obj in lang_network.nodes.items():
                        node_parts = node_id.split('_')
                        node_group = node_parts[0] # e.g., 'chars', 'words', 'semantic'
                        if node_group == "semantic": # Handle "semantic_concepts_X"
                            node_group = "semantic_concepts"
                            node_index = int(node_parts[-1]) # Index is the last part
                        elif len(node_parts) > 1:
                            node_index = int(node_parts[-1]) # Index is the last part for "chars_X", "words_X"
                        else:
                            # Should not happen with our current node naming, but good to have a fallback
                            st.warning(f"Unexpected node_id format: {node_id}. Skipping activation for this node.")
                            continue
                            
                        activated = False

                        if node_group == 'chars':
                            # Activate a few char nodes based on text length or a simple hash
                            if len(text_input) > 0 and node_index == (len(text_input) % len(linguistic_features['chars'])):
                                activated = True
                        elif node_group == 'words':
                            # Activate if a word hash matches the node index (simplified)
                            # This is a very basic way to get some word-related activation
                            for word in words_in_text:
                                if node_index == (hash(word) % len(linguistic_features['words'])):
                                    activated = True
                                    break # Activate per node once
                        elif node_group == 'semantic_concepts':
                            # Activate a couple of semantic nodes based on text properties
                            if len(words_in_text) > 3 and node_index < 2: # Activate first 2 if text is non-trivial
                                activated = True
                            elif 'music' in words_in_text and node_index == (len(linguistic_features['semantic_concepts']) -1):
                                activated = True # Specific activation for a keyword
                        
                        if activated:
                            node_obj.amplitude = 1.0
                            node_obj.phase = np.random.uniform(0, 2 * np.pi) # Randomize phase on activation
                            activated_in_lang_network += 1
                        else:
                            node_obj.amplitude = 0.001 # Dampen other nodes
                    
                    st.write(f"Activated {activated_in_lang_network} nodes in language network with new logic.")
                    # --- End of New Activation Logic ---

                    # Log active nodes from encoding (This warning is now expected as we changed activation strategy)
                    # if active_node_ids_from_encoding:
                    #     st.write(f"Nodes directly activated by encoder (set to Amp 1.0):** {len(active_node_ids_from_encoding)} nodes")
                    #     # st.write(list(active_node_ids_from_encoding)[:5]) # Show a few
                    # else:
                    #     st.warning("No nodes were directly activated by the text encoder based on its output keys.")

                    # Simulate network dynamics
                    simulation_steps = 75 # Slightly increase simulation steps
                    for i_step in range(simulation_steps):
                        lang_network.step(0.01)
                        # if i_step % 10 == 0: # Optional: log amplitudes during simulation
                        #     amps = {nid: n.amplitude for nid, n in lang_network.nodes.items() if n.amplitude > 0.05}
                        #     if amps:
                        #         st.write(f"Amplitudes at step {i_step}: {amps}")
                    
                    st.success("Text processed by language network.")

                    # Log final amplitudes in lang_network before sonification
                    final_lang_network_amplitudes = {nid: n.amplitude for nid, n in lang_network.nodes.items() if n.amplitude > 0.05}
                    if final_lang_network_amplitudes:
                        st.write("**Final Amplitudes in Language Network (actives > 0.05):**")
                        st.json(final_lang_network_amplitudes)
                    else:
                        st.warning("Language network shows no significant activity after simulation.")

                    # 3. Generate selected outputs
                    if "Audio Pattern" in output_modalities:
                        audio_pattern = self._generate_audio_from_network_state(lang_network, duration=3.0)
                        st.markdown("**Generated Audio Pattern:**")
                        st.audio(audio_pattern, sample_rate=22050)
                    
                    if "Visual Pattern (Network State)" in output_modalities:
                        st.markdown("**Network State Visualization:**")
                        # Simplified visualization: bar chart of node amplitudes
                        node_ids = list(lang_network.nodes.keys())[:20] # Limit for clarity
                        amplitudes = [lang_network.nodes[nid].amplitude for nid in node_ids]
                        frequencies = [lang_network.nodes[nid].frequency for nid in node_ids]
                        
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.bar(range(len(node_ids)), amplitudes, color='skyblue')
                        ax.set_xticks(range(len(node_ids)))
                        ax.set_xticklabels([f"{nid.split('_')[0]}\n({int(freq)}Hz)" for nid, freq in zip(node_ids, frequencies)], rotation=45, ha="right", fontsize=8)
                        ax.set_ylabel("Amplitude")
                        ax.set_title("Language Network Node Amplitudes (Post-Text Processing)")
                        plt.tight_layout()
                        st.pyplot(fig)

                    if "Textual Description (Interpretation)" in output_modalities:
                        # Conceptual: interpret network state to text
                        interpretation = self._interpret_network_state_to_text(lang_network, original_text=text_input)
                        st.markdown("**Generated Textual Interpretation:**")
                        st.write(interpretation)
                    # st.success("Multimodal processing complete (placeholder results).")
        
        elif input_modality == "Image (coming soon)":
            # uploaded_image = st.file_uploader("Upload an Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="mm_image_upload")
            # if uploaded_image and st.button("Process Image and Generate Outputs", key="mm_process_image"):
            #     # Process image...
            st.warning("Image processing functionality will be implemented soon.")
        
        elif input_modality == "Audio (coming soon)":
            # uploaded_audio = st.file_uploader("Upload Audio (WAV)", type=["wav"], key="mm_audio_upload")
            # if uploaded_audio and st.button("Process Audio and Generate Outputs", key="mm_process_audio"):
            #     # Process audio...
            st.warning("Audio input processing functionality will be implemented soon.")

    # Helper methods for multimodal demo - to be defined or expanded
    def _create_simplified_language_network_for_demo(self) -> ResonantNetwork:
        # Based on MultiModalResonantSystem._create_language_network but simplified
        network = AcceleratedResonantNetwork(name="mm_lang_demo_network", backend=st.session_state.get('backend_type', BackendType.CPU)) # Use selected backend, default to CPU
        network.coupling_strength = 1.8 # Slightly increase coupling
        network.global_damping = 0.12 # Slightly decrease damping to retain energy longer

        # Fewer, broader categories for a simpler demo
        # Make this accessible for the activation logic
        self.linguistic_feature_bands_for_mm_demo = {
            'chars': np.linspace(100, 300, 10),    
            'words': np.linspace(300, 700, 15),    
            'semantic_concepts': np.linspace(700, 1200, 10) 
        }
        node_count = 0
        for feature_type, frequencies in self.linguistic_feature_bands_for_mm_demo.items():
            for i, freq in enumerate(frequencies):
                node = ResonantNode(
                    node_id=f"{feature_type}_{i}", # Node IDs TextPatternEncoder might generate
                    frequency=freq + np.random.uniform(-5,5), # Add some jitter
                    phase=np.random.uniform(0, 2*np.pi),
                    amplitude=0.1 # Start low
                )
                network.add_node(node)
                node_count +=1
        
        # Simple connections: within groups and some cross-group
        all_nodes = list(network.nodes.keys())
        for i in range(node_count):
            for j in range(i + 1, node_count):
                if np.random.rand() < 0.1: # Sparse connections
                     # Connect based on feature type similarity (simplified)
                    if all_nodes[i].split('_')[0] == all_nodes[j].split('_')[0]: # within same group
                        network.connect(all_nodes[i], all_nodes[j], weight=np.random.uniform(0.2, 0.5))
                    else: # between different groups
                        network.connect(all_nodes[i], all_nodes[j], weight=np.random.uniform(0.05, 0.20)) # Slightly increased inter-group connection possibility
        return network

    def _generate_audio_from_network_state(self, network: ResonantNetwork, duration: float = 3.0, sample_rate: int = 22050) -> np.ndarray:
        # Simple sonification: active nodes contribute their frequency
        # This is a basic approach; more sophisticated methods could map phase, amplitude dynamics etc.
        
        # Create a temporary network snapshot for generation if using AcceleratedResonantNetwork
        # to avoid issues with ongoing simulation state, or ensure network.generate_signals is safe.
        temp_gen_network = AcceleratedResonantNetwork(name="temp_audio_gen", backend=network.backend)
        temp_gen_network.coupling_strength = 0 
        temp_gen_network.global_damping = 0.01

        active_nodes_for_sonification = []
        min_activation_threshold = 0.08 # Lowered threshold slightly 

        st.write("--- Debug Sonification --- ") 
        st.write(f"Target Sonification Duration: {duration}s, Sample Rate: {sample_rate}Hz")
        lang_network_active_nodes_for_debug = {nid: (n.frequency, n.amplitude, n.phase) for nid, n in network.nodes.items() if n.amplitude > min_activation_threshold}
        st.write(f"Nodes from lang_network with Amp > {min_activation_threshold} before sonification processing:")
        if lang_network_active_nodes_for_debug:
            st.json(lang_network_active_nodes_for_debug)
        else:
            st.write("None")

        for node_id, original_node in network.nodes.items():
            if original_node.amplitude > min_activation_threshold: 
                # Create a new node in the temp network with the current state
                # Boost amplitude for sonification if it's low but active
                sonification_amplitude = original_node.amplitude
                if sonification_amplitude < 0.5: # If active but low, boost it a bit for audibility
                    sonification_amplitude = max(sonification_amplitude * 2.0, 0.5) # Ensure it reaches at least 0.5
                sonification_amplitude = min(sonification_amplitude, 1.0) # Cap at 1.0

                new_node = ResonantNode(
                    node_id=original_node.node_id,
                    frequency=original_node.frequency,
                    phase=original_node.phase, # Preserve phase
                    amplitude=sonification_amplitude 
                )
                temp_gen_network.add_node(new_node)
                active_nodes_for_sonification.append((original_node.node_id, original_node.frequency, original_node.amplitude, sonification_amplitude))
        
        # st.write(f"Nodes for sonification (ID, Orig Amp, Sonif Amp): {active_nodes_for_sonification}")
        if active_nodes_for_sonification:
            st.write(f"**Nodes chosen for sonification in temp_gen_network (ID, Freq, Orig Amp, Sonif Amp):**")
            st.json(active_nodes_for_sonification)
        else:
            st.warning("No nodes met criteria for sonification in temp_gen_network.")

        if not temp_gen_network.nodes: 
            st.warning("temp_gen_network has no nodes. Audio will be silent.")

        audio_output = temp_gen_network.generate_signals(duration, sample_rate)
        
        if audio_output.size == 0:
             return np.zeros(int(duration * sample_rate))

        # Normalize final audio
        max_val = np.max(np.abs(audio_output))
        if max_val > 0:
            audio_output = audio_output / max_val * 0.7 # headroom
        else:
            st.warning("Generated audio was all zeros before final normalization.")
        
        st.write("--- End Debug Sonification --- ")
        return audio_output

    def _interpret_network_state_to_text(self, network: ResonantNetwork, original_text: str = "") -> str:
        # Simplified interpretation based on dominant frequencies and active groups
        active_nodes = sorted([node for node_id, node in network.nodes.items() if node.amplitude > 0.2], key=lambda x: x.amplitude, reverse=True)
        
        if not active_nodes:
            return "The network showed minimal resonant activity."

        interpretation_parts = [f"After processing '{original_text[:30]}...', the network shows:"]
        
        top_n = min(len(active_nodes), 5)
        interpretation_parts.append(f"\n**Dominant Resonances (Top {top_n}):**")
        for node in active_nodes[:top_n]:
            feature_group = node.node_id.split('_')[0]
            interpretation_parts.append(f"  - Strong activity in '{feature_group}' band (node: {node.node_id}, Freq: {node.frequency:.2f} Hz, Amp: {node.amplitude:.2f})")

        # Count active nodes per group
        group_activity = {}
        for node in active_nodes:
            group = node.node_id.split('_')[0]
            group_activity[group] = group_activity.get(group, 0) + 1
        
        if group_activity:
            interpretation_parts.append("\n**Overall Thematic Resonance:**")
            for group, count in sorted(group_activity.items(), key=lambda item: item[1], reverse=True):
                interpretation_parts.append(f"  - '{group.capitalize()}' concepts appear to be significant ({count} active nodes).")
        
        # Example: a very basic semantic guess
        if 'semantic_concepts' in group_activity and group_activity['semantic_concepts'] > 1:
            interpretation_parts.append("\nThe pattern suggests engagement with abstract semantic representations.")
        elif 'words' in group_activity and group_activity['words'] > 3:
            interpretation_parts.append("\nThe resonance pattern is strongly word-focused.")

        return "\n".join(interpretation_parts)

    def render_emotional_resonance_demo(self):
        """Render the Emotional Resonance Engine demo."""
        # Initialize the engine
        if 'emotional_engine' not in st.session_state:
            st.session_state.emotional_engine = EmotionalResonanceEngine()
            
        engine = st.session_state.emotional_engine
        
        # Create sub-tabs for different demos
        emotion_tabs = st.tabs([
            "🎭 Emotion Explorer",
            "🤝 Empathetic Response",
            "🎨 Emotional Journey",
            "🎵 Analyze Audio",
            "🌍 Cultural Context"
        ])
        
        with emotion_tabs[0]:  # Emotion Explorer
            st.subheader("Explore Emotional Signatures")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Emotion selector
                selected_emotion = st.selectbox(
                    "Select Emotion",
                    options=[e for e in EmotionCategory],
                    format_func=lambda x: x.value.title()
                )
                
                # Intensity slider
                intensity = st.slider(
                    "Emotional Intensity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.1
                )
                
                # Duration
                duration = st.slider(
                    "Duration (seconds)",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
                
                # Generate button
                if st.button("🎵 Generate Emotion", key="gen_emotion"):
                    with st.spinner("Creating emotional resonance..."):
                        # Create emotional network
                        network = engine.create_emotional_network(
                            selected_emotion,
                            intensity
                        )
                        
                        # Generate audio
                        audio = engine.generate_empathetic_response(
                            selected_emotion,
                            "matching",
                            duration
                        )
                        
                        st.session_state.emotional_audio = audio
                        st.session_state.current_emotion = selected_emotion
                        
            with col2:
                # Display emotion signature
                if selected_emotion:
                    signature = engine.emotion_signatures[selected_emotion]
                    
                    # Create info display
                    st.markdown("### Emotional Signature")
                    
                    # Color display
                    color = signature.color_association
                    st.markdown(
                        f'<div style="background-color: rgb{color}; '
                        f'width: 100%; height: 50px; border-radius: 10px; '
                        f'margin-bottom: 20px;"></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Base Frequency", f"{signature.base_frequency:.1f} Hz")
                        st.metric("Energy Level", f"{signature.energy_level:.2f}")
                    with col_b:
                        st.metric("Tempo Range", f"{signature.tempo_range[0]}-{signature.tempo_range[1]} BPM")
                        st.metric("Valence", f"{signature.valence:+.2f}")
                    with col_c:
                        st.metric("Phase Coherence", f"{signature.phase_coherence:.2f}")
                        st.metric("Mode", signature.modal_preference.title())
                    
                    # Harmonic series visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    harmonics = range(1, len(signature.harmonic_series) + 1)
                    ax.bar(harmonics, signature.harmonic_series, 
                          color=[color[i]/255 for i in range(3)] + [0.7])
                    ax.set_xlabel("Harmonic Number")
                    ax.set_ylabel("Relative Strength")
                    ax.set_title(f"Harmonic Series for {signature.name}")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            # Audio player
            if 'emotional_audio' in st.session_state and st.session_state.emotional_audio is not None:
                st.markdown("### Generated Emotional Audio")
                audio_html = self.create_audio_player(st.session_state.emotional_audio)
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Waveform visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                time_axis = np.arange(len(st.session_state.emotional_audio)) / 44100
                ax.plot(time_axis, st.session_state.emotional_audio, linewidth=0.5)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Waveform - {st.session_state.current_emotion.value.title()}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
        with emotion_tabs[1]:  # Empathetic Response
            st.subheader("Generate Empathetic Responses")
            st.markdown("""
            Select an input emotion and response type to generate an empathetic 
            musical response that either matches, complements, or balances the emotion.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                input_emotion = st.selectbox(
                    "Input Emotion",
                    options=[e for e in EmotionCategory],
                    format_func=lambda x: x.value.title(),
                    key="input_emotion"
                )
                
                response_type = st.radio(
                    "Response Type",
                    options=["matching", "complementary", "balancing"],
                    help="""
                    - **Matching**: Reflects the same emotion back
                    - **Complementary**: Responds with a supportive emotion
                    - **Balancing**: Provides emotional equilibrium
                    """
                )
                
            with col2:
                st.markdown("### Response Mapping")
                
                # Show what the response will be
                if response_type == "matching":
                    response_emotion = input_emotion
                    st.info(f"Response will match: **{input_emotion.value.title()}**")
                elif response_type == "complementary":
                    complementary_map = {
                        EmotionCategory.JOY: EmotionCategory.LOVE,
                        EmotionCategory.SADNESS: EmotionCategory.CALM,
                        EmotionCategory.ANGER: EmotionCategory.CALM,
                        EmotionCategory.FEAR: EmotionCategory.LOVE,
                        EmotionCategory.EXCITEMENT: EmotionCategory.JOY,
                    }
                    response_emotion = complementary_map.get(input_emotion, EmotionCategory.CALM)
                    st.info(f"Response will be: **{response_emotion.value.title()}**")
                else:  # balancing
                    st.info("Response will balance with opposite valence emotion")
                
            if st.button("🎵 Generate Response", key="gen_response"):
                with st.spinner("Generating empathetic response..."):
                    response_audio = engine.generate_empathetic_response(
                        input_emotion,
                        response_type,
                        duration=4.0
                    )
                    
                    st.session_state.response_audio = response_audio
                    
                    # Display both waveforms
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                    
                    # Input emotion (generate for comparison)
                    input_audio = engine.generate_empathetic_response(
                        input_emotion, "matching", duration=4.0
                    )
                    
                    time_axis = np.arange(len(input_audio)) / 44100
                    
                    # Input waveform
                    ax1.plot(time_axis, input_audio, linewidth=0.5, alpha=0.7)
                    ax1.set_title(f"Input: {input_emotion.value.title()}")
                    ax1.set_ylabel("Amplitude")
                    ax1.grid(True, alpha=0.3)
                    
                    # Response waveform
                    ax2.plot(time_axis, response_audio, linewidth=0.5, alpha=0.7)
                    ax2.set_title(f"Response: {response_type.title()}")
                    ax2.set_xlabel("Time (seconds)")
                    ax2.set_ylabel("Amplitude")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Audio player for response
                    st.markdown("### Empathetic Response Audio")
                    audio_html = self.create_audio_player(response_audio)
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
        with emotion_tabs[2]:  # Emotional Journey
            st.subheader("Create an Emotional Journey")
            st.markdown("""
            Design a journey through multiple emotions with smooth transitions.
            """)
            
            # Journey builder
            if 'emotion_journey' not in st.session_state:
                st.session_state.emotion_journey = []
                
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Add emotion to journey
                new_emotion = st.selectbox(
                    "Add Emotion",
                    options=[e for e in EmotionCategory],
                    format_func=lambda x: x.value.title(),
                    key="journey_emotion"
                )
                
                emotion_duration = st.slider(
                    "Duration (seconds)",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    key="journey_duration"
                )
                
                if st.button("➕ Add to Journey"):
                    st.session_state.emotion_journey.append(
                        (new_emotion, emotion_duration)
                    )
                    
            with col2:
                # Display current journey
                st.markdown("### Current Journey")
                if st.session_state.emotion_journey:
                    total_duration = sum(d for _, d in st.session_state.emotion_journey)
                    st.metric("Total Duration", f"{total_duration:.1f}s")
                    
                    for i, (emotion, duration) in enumerate(st.session_state.emotion_journey):
                        st.text(f"{i+1}. {emotion.value.title()} ({duration}s)")
                        
                    if st.button("🗑️ Clear Journey"):
                        st.session_state.emotion_journey = []
                        st.experimental_rerun()
                else:
                    st.info("No emotions added yet")
                    
            # Generate journey
            if st.session_state.emotion_journey and st.button("🎵 Generate Journey", key="gen_journey"):
                with st.spinner("Creating emotional journey..."):
                    journey_audio = engine.create_emotional_journey(
                        st.session_state.emotion_journey,
                        transition_time=1.0
                    )
                    
                    st.session_state.journey_audio = journey_audio
                    
                    # Visualize journey
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create color bands for each emotion
                    current_time = 0
                    for emotion, duration in st.session_state.emotion_journey:
                        color = engine.get_emotion_color(emotion)
                        color_norm = [c/255 for c in color]
                        
                        ax.axvspan(current_time, current_time + duration,
                                 color=color_norm, alpha=0.3,
                                 label=emotion.value.title())
                        
                        # Add emotion label
                        ax.text(current_time + duration/2, 0.9,
                               emotion.value.title(),
                               ha='center', va='center',
                               transform=ax.get_xaxis_transform())
                        
                        current_time += duration
                    
                    # Plot waveform
                    time_axis = np.arange(len(journey_audio)) / 44100
                    ax.plot(time_axis, journey_audio, linewidth=0.5, color='black', alpha=0.7)
                    
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Amplitude")
                    ax.set_title("Emotional Journey")
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, time_axis[-1])
                    
                    st.pyplot(fig)
                    
                    # Audio player
                    st.markdown("### Journey Audio")
                    audio_html = self.create_audio_player(journey_audio)
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
        with emotion_tabs[3]:  # Analyze Audio
            st.subheader("Analyze Emotional Content")
            st.markdown("""
            Upload an audio file or use generated audio to analyze its emotional content.
            """)
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'ogg'],
                key="emotion_upload"
            )
            
            audio_to_analyze = None
            
            if uploaded_file is not None:
                # Read uploaded file
                audio_data = uploaded_file.read()
                # For simplicity, assume WAV format
                # In production, you'd want to handle multiple formats
                import io
                from scipy.io import wavfile
                
                try:
                    sr, audio_to_analyze = wavfile.read(io.BytesIO(audio_data))
                    # Convert to mono if stereo
                    if len(audio_to_analyze.shape) > 1:
                        audio_to_analyze = audio_to_analyze.mean(axis=1)
                    # Normalize
                    audio_to_analyze = audio_to_analyze / np.max(np.abs(audio_to_analyze))
                except:
                    st.error("Error reading audio file. Please upload a valid WAV file.")
                    
            elif 'emotional_audio' in st.session_state:
                if st.button("Use Generated Audio"):
                    audio_to_analyze = st.session_state.emotional_audio
                    
            if audio_to_analyze is not None:
                with st.spinner("Analyzing emotional content..."):
                    # Analyze emotion
                    emotion_scores = engine.analyze_emotional_content(audio_to_analyze)
                    
                    # Sort by score
                    sorted_emotions = sorted(
                        emotion_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Display results
                    st.markdown("### Detected Emotions")
                    
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    emotions = [e[0].value.title() for e in sorted_emotions[:8]]
                    scores = [e[1] for e in sorted_emotions[:8]]
                    colors = [engine.get_emotion_color(e[0]) for e in sorted_emotions[:8]]
                    colors_norm = [[c/255 for c in color] for color in colors]
                    
                    bars = ax.bar(emotions, scores, color=colors_norm)
                    ax.set_ylabel("Probability")
                    ax.set_title("Emotional Content Analysis")
                    ax.set_ylim(0, max(scores) * 1.2)
                    
                    # Add value labels on bars
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{score:.2%}',
                               ha='center', va='bottom')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Top emotions summary
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (emotion, score) in enumerate(sorted_emotions[:3]):
                        with [col1, col2, col3][i]:
                            color = engine.get_emotion_color(emotion)
                            st.markdown(
                                f'<div style="background-color: rgb{color}; '
                                f'padding: 20px; border-radius: 10px; text-align: center;">'
                                f'<h3 style="margin: 0;">{emotion.value.title()}</h3>'
                                f'<h2 style="margin: 10px 0;">{score:.1%}</h2>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
        with emotion_tabs[4]:  # Cultural Context
            st.subheader("Cultural Emotion Expression")
            st.markdown("""
            Explore how different cultures express emotions with varying intensities.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                cultural_context = st.selectbox(
                    "Cultural Context",
                    options=["None", "western", "eastern", "latin", "nordic"],
                    key="cultural_context"
                )
                
                selected_culture_emotion = st.selectbox(
                    "Emotion",
                    options=[e for e in EmotionCategory],
                    format_func=lambda x: x.value.title(),
                    key="culture_emotion"
                )
                
                base_intensity = st.slider(
                    "Base Intensity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    key="culture_intensity"
                )
                
            with col2:
                # Show cultural modifier if applicable
                if cultural_context != "None" and cultural_context in engine.cultural_modifiers:
                    modifiers = engine.cultural_modifiers[cultural_context]
                    
                    st.markdown("### Cultural Modifiers")
                    
                    # Display all modifiers for this culture
                    for emotion, modifier in modifiers.items():
                        if emotion == selected_culture_emotion:
                            st.success(f"**{emotion.value.title()}**: {modifier:.1f}x (selected)")
                        else:
                            st.info(f"{emotion.value.title()}: {modifier:.1f}x")
                            
                    # Calculate effective intensity
                    if selected_culture_emotion in modifiers:
                        effective_intensity = base_intensity * modifiers[selected_culture_emotion]
                        st.metric(
                            "Effective Intensity",
                            f"{effective_intensity:.2f}",
                            f"{(effective_intensity - base_intensity):+.2f}"
                        )
                        
            # Generate comparison
            if st.button("🎵 Generate Comparison", key="gen_culture"):
                with st.spinner("Generating cultural comparison..."):
                    # Generate without cultural context
                    network_base = engine.create_emotional_network(
                        selected_culture_emotion,
                        base_intensity
                    )
                    audio_base = engine.generate_empathetic_response(
                        selected_culture_emotion,
                        "matching",
                        duration=3.0
                    )
                    
                    # Generate with cultural context
                    if cultural_context != "None":
                        network_cultural = engine.create_emotional_network(
                            selected_culture_emotion,
                            base_intensity,
                            cultural_context
                        )
                        audio_cultural = engine.generate_empathetic_response(
                            selected_culture_emotion,
                            "matching",
                            duration=3.0
                        )
                    else:
                        audio_cultural = audio_base
                        
                    # Visualize comparison
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                    
                    time_axis = np.arange(len(audio_base)) / 44100
                    
                    # Base audio
                    ax1.plot(time_axis, audio_base, linewidth=0.5, alpha=0.7)
                    ax1.set_title(f"Base Expression - {selected_culture_emotion.value.title()}")
                    ax1.set_ylabel("Amplitude")
                    ax1.grid(True, alpha=0.3)
                    
                    # Cultural audio
                    ax2.plot(time_axis, audio_cultural, linewidth=0.5, alpha=0.7)
                    ax2.set_title(f"Cultural Expression ({cultural_context})")
                    ax2.set_xlabel("Time (seconds)")
                    ax2.set_ylabel("Amplitude")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Audio players
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Base Expression")
                        audio_html = self.create_audio_player(audio_base)
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown(f"### {cultural_context.title() if cultural_context != 'None' else 'Base'} Expression")
                        audio_html = self.create_audio_player(audio_cultural)
                        st.markdown(audio_html, unsafe_allow_html=True)


def resonance_field_4d_demo():
    """Demo for 4D Resonance Fields"""
    st.header("🌊 4D Resonance Fields")
    st.write("Explore wave propagation in 3D space + time!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Field Parameters")
        
        # Field dimensions
        field_size = st.slider("Field Size", 10, 50, 20)
        wave_speed = st.slider("Wave Speed", 10.0, 500.0, 100.0)
        
        # Boundary conditions
        boundary = st.selectbox(
            "Boundary Condition",
            ["ABSORBING", "REFLECTING", "PERIODIC", "RADIATING"]
        )
        
        # Add nodes
        st.subheader("Add Spatial Nodes")
        num_nodes = st.number_input("Number of Nodes", 1, 10, 3)
        
        if st.button("Create Field"):
            with st.spinner("Creating 4D resonance field..."):
                # Create field
                field = ResonanceField4D(
                    dimensions=(field_size, field_size, field_size),
                    resolution=1.0,
                    wave_speed=wave_speed,
                    boundary_condition=BoundaryCondition[boundary]
                )
                
                # Add random nodes
                for i in range(num_nodes):
                    pos = np.random.rand(3) * field_size
                    freq = np.random.uniform(1.0, 5.0)
                    node = SpatialResonantNode(
                        f"node_{i}",
                        position=pos,
                        frequency=freq,
                        amplitude=1.0,
                        radiation_pattern="omnidirectional"
                    )
                    field.add_spatial_node(node)
                
                # Store in session state
                st.session_state.resonance_field = field
                
    with col2:
        if hasattr(st.session_state, 'resonance_field'):
            field = st.session_state.resonance_field
            
            # Simulate steps
            steps = st.slider("Simulation Steps", 0, 200, 50)
            
            if st.button("Simulate"):
                progress_bar = st.progress(0)
                
                # Run simulation
                for i in range(steps):
                    field.step()
                    progress_bar.progress((i + 1) / steps)
                
                # Visualize results
                st.subheader("Field Visualization")
                
                # Get slices
                slice_axis = st.selectbox("Slice Axis", ["x", "y", "z"])
                slice_idx = st.slider("Slice Position", 0, field.dimensions[0]-1, field.dimensions[0]//2)
                
                field_slice = field.extract_slice(slice_axis, slice_idx)
                
                # Plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Amplitude field
                im1 = ax1.imshow(field_slice, cmap='seismic', vmin=-1, vmax=1)
                ax1.set_title(f"Amplitude Field ({slice_axis}={slice_idx})")
                plt.colorbar(im1, ax=ax1)
                
                # Field statistics
                stats = field.get_field_statistics()
                ax2.text(0.1, 0.8, f"Total Energy: {stats['total_energy']:.2e}", fontsize=12)
                ax2.text(0.1, 0.6, f"Max Amplitude: {stats['max_amplitude']:.3f}", fontsize=12)
                ax2.text(0.1, 0.4, f"Coherence: {stats['coherence']:.3f}", fontsize=12)
                ax2.text(0.1, 0.2, f"Entropy: {stats['entropy']:.2f}", fontsize=12)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.axis('off')
                ax2.set_title("Field Statistics")
                
                st.pyplot(fig)
                plt.close()


def collective_intelligence_demo():
    """Demo for Collective Intelligence"""
    st.header("🧠 Collective Intelligence")
    st.write("Watch multiple networks work together to make decisions!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Collective Setup")
        
        # Communication mode
        comm_mode = st.selectbox(
            "Communication Mode",
            ["PHASE_COUPLING", "FREQUENCY_MODULATION", "DIRECT"]
        )
        
        # Consensus method
        consensus = st.selectbox(
            "Consensus Method",
            ["SYNCHRONIZATION", "VOTING", "EMERGENCE", "HIERARCHICAL"]
        )
        
        # Number of networks
        num_networks = st.slider("Number of Networks", 2, 8, 4)
        
        if st.button("Create Collective"):
            with st.spinner("Creating collective intelligence..."):
                # Create collective
                collective = CollectiveIntelligence(
                    name="demo_collective",
                    communication_mode=CommunicationMode[comm_mode],
                    consensus_method=ConsensusMethod[consensus]
                )
                
                # Add specialized networks
                roles = ["sensor", "processor", "memory", "decision", "coordinator", "analyzer", "integrator", "output"]
                
                for i in range(num_networks):
                    # Create network
                    network = ResonantNetwork(f"network_{i}")
                    
                    # Add nodes
                    for j in range(5):
                        node = ResonantNode(
                            f"n{i}_{j}",
                            frequency=1.0 + i*0.5 + j*0.1,
                            damping=0.05
                        )
                        network.add_node(node)
                    
                    # Connect nodes (ring topology)
                    for j in range(5):
                        network.connect(f"n{i}_{j}", f"n{i}_{(j+1)%5}")
                    
                    # Create role
                    role = NetworkRole(
                        name=roles[i % len(roles)],
                        frequency_band=(1.0 + i*0.5, 2.0 + i*0.5),
                        influence_weight=1.0 + i*0.2,
                        receptivity=0.8,
                        specialization=roles[i % len(roles)]
                    )
                    
                    collective.add_network(f"net_{i}", network, role)
                
                # Connect networks
                for i in range(num_networks-1):
                    collective.connect_networks(f"net_{i}", f"net_{i+1}")
                
                st.session_state.collective = collective
                
    with col2:
        if hasattr(st.session_state, 'collective'):
            collective = st.session_state.collective
            
            st.subheader("Collective Dynamics")
            
            # Run simulation
            sim_steps = st.slider("Simulation Steps", 10, 500, 100)
            
            if st.button("Run Collective"):
                progress_bar = st.progress(0)
                sync_history = []
                
                for i in range(sim_steps):
                    collective.step(0.01)
                    if i % 10 == 0:
                        sync_history.append(collective.synchronization_index)
                    progress_bar.progress((i + 1) / sim_steps)
                
                # Visualize
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Synchronization over time
                ax1.plot(sync_history, linewidth=2)
                ax1.set_xlabel("Time Steps (x10)")
                ax1.set_ylabel("Synchronization Index")
                ax1.set_title("Collective Synchronization")
                ax1.grid(True, alpha=0.3)
                
                # Network states
                viz_data = collective.visualize_collective_state()
                networks = list(viz_data['networks'].keys())
                freqs = [viz_data['networks'][n]['mean_frequency'] for n in networks]
                syncs = [viz_data['networks'][n]['synchronization'] for n in networks]
                
                x = np.arange(len(networks))
                width = 0.35
                
                ax2.bar(x - width/2, freqs, width, label='Mean Frequency')
                ax2.bar(x + width/2, syncs, width, label='Synchronization')
                ax2.set_xlabel("Network")
                ax2.set_ylabel("Value")
                ax2.set_title("Individual Network States")
                ax2.set_xticks(x)
                ax2.set_xticklabels(networks, rotation=45)
                ax2.legend()
                ax2.grid(True, axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Decision making
            st.subheader("Collective Decision Making")
            
            options = st.text_area("Options (one per line)", "Option A\nOption B\nOption C").split('\n')
            
            if st.button("Make Decision"):
                decision = collective.make_collective_decision(options)
                st.success(f"Collective Decision: **{decision}**")
                
                # Show decision history
                if collective.decision_history:
                    latest = collective.decision_history[-1]
                    st.write("Decision Details:")
                    st.json(latest)


def evolutionary_resonance_demo():
    """Demo for Evolutionary Resonance"""
    st.header("🧬 Evolutionary Resonance")
    st.write("Evolve networks to optimize for specific fitness criteria!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Evolution Parameters")
        
        # Population parameters
        pop_size = st.slider("Population Size", 10, 100, 30)
        
        # Fitness metric
        fitness = st.selectbox(
            "Fitness Metric",
            ["HARMONY", "EFFICIENCY", "INFORMATION", "ADAPTABILITY", "CREATIVITY", "MEMORY"]
        )
        
        # Evolution parameters
        mutation_rate = st.slider("Mutation Rate", 0.0, 0.5, 0.15)
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.7)
        
        # Number of generations
        num_generations = st.slider("Generations", 5, 50, 20)
        
        if st.button("Start Evolution"):
            with st.spinner("Initializing evolution..."):
                # Create evolutionary system
                evo = EvolutionaryResonance(
                    population_size=pop_size,
                    fitness_metric=FitnessMetric[fitness],
                    mutation_rate=mutation_rate,
                    crossover_rate=crossover_rate,
                    speciation_threshold=0.3
                )
                
                # Initialize population
                evo.initialize_population()
                
                st.session_state.evo_system = evo
                st.session_state.generation = 0
                
    with col2:
        if hasattr(st.session_state, 'evo_system'):
            evo = st.session_state.evo_system
            
            st.subheader("Evolution Progress")
            
            if st.button("Evolve"):
                progress_bar = st.progress(0)
                
                best_fitness_history = []
                species_count_history = []
                
                for gen in range(num_generations):
                    evo.evolve_generation()
                    
                    best = evo.get_best_genome()
                    best_fitness_history.append(best.fitness)
                    species_count_history.append(len(evo.species))
                    
                    progress_bar.progress((gen + 1) / num_generations)
                
                # Visualize evolution
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Fitness evolution
                ax1.plot(best_fitness_history, linewidth=2, color='green')
                ax1.set_xlabel("Generation")
                ax1.set_ylabel("Best Fitness")
                ax1.set_title("Fitness Evolution")
                ax1.grid(True, alpha=0.3)
                
                # Species diversity
                ax2.plot(species_count_history, linewidth=2, color='blue')
                ax2.set_xlabel("Generation")
                ax2.set_ylabel("Number of Species")
                ax2.set_title("Species Diversity")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show best network
                st.subheader("Best Evolved Network")
                
                best_genome = evo.get_best_genome()
                st.write(f"**Fitness:** {best_genome.fitness:.3f}")
                st.write(f"**Nodes:** {len(best_genome.node_genes)}")
                st.write(f"**Connections:** {len(best_genome.connection_genes)}")
                
                # Species information
                st.subheader("Species Information")
                species_info = evo.get_species_info()
                
                for species_id, info in species_info.items():
                    with st.expander(f"{species_id} ({info['size']} members)"):
                        st.write(f"Mean Fitness: {info['mean_fitness']:.3f}")
                        st.write(f"Max Fitness: {info['max_fitness']:.3f}")
                        st.write(f"Age: {info['age']} generations")
                        st.write(f"Stagnation: {info['stagnation']} generations")


if __name__ == "__main__":
    demo = SynthNNDemo()
    demo.main() 