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

# Import SynthNN components
from synthnn.core import ResonantNode, ResonantNetwork, SignalProcessor, UniversalPatternCodec
from synthnn.performance import AcceleratedResonantNetwork, BackendManager, BackendType

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
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎼 Music Generation", 
            "📊 Network Visualization", 
            "🔬 Analysis",
            "🎮 Interactive Control"
        ])
        
        with tab1:
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
                
        with tab2:
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
                
            # Synchronization metric
            if st.session_state.network_history:
                sync_values = [s['synchronization'] for s in st.session_state.network_history]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(sync_values, linewidth=2)
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Synchronization")
                ax.set_title("Network Synchronization Over Time")
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
        with tab3:
            st.header("Audio Analysis")
            
            if st.session_state.audio_buffer is not None:
                # Analyze audio
                spectrum, fundamental, coherence = self.analyze_audio(
                    st.session_state.audio_buffer
                )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Fundamental Frequency", f"{fundamental:.1f} Hz")
                    
                with col2:
                    st.metric("Average Coherence", f"{np.mean(coherence):.3f}")
                    
                with col3:
                    st.metric("Network Nodes", len(network.nodes))
                    
                # Spectrum plot
                st.subheader("Frequency Spectrum")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                freqs = spectrum['frequencies']
                magnitudes = spectrum['magnitudes']
                
                # Find peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)
                
                ax.plot(freqs, magnitudes, linewidth=1.5, label='Spectrum')
                ax.plot(freqs[peaks], magnitudes[peaks], 'ro', markersize=8, 
                       label=f'Peaks ({len(peaks)})')
                
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Magnitude")
                ax.set_title("Frequency Spectrum Analysis")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_xlim(0, 2000)  # Focus on lower frequencies
                st.pyplot(fig)
                
                # Phase coherence over time
                st.subheader("Phase Coherence Evolution")
                fig, ax = plt.subplots(figsize=(10, 4))
                time_axis = np.linspace(0, len(st.session_state.audio_buffer) / 44100, 
                                      len(coherence))
                ax.plot(time_axis, coherence, linewidth=2)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Phase Coherence")
                ax.set_title("Phase Coherence Over Time")
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                
            else:
                st.info("Generate audio first to see analysis")
                
        with tab4:
            st.header("Interactive Control")
            
            st.markdown("""
            Apply external stimuli to specific nodes and observe how the network responds.
            This demonstrates the network's ability to process and integrate external signals.
            """)
            
            # Node selection for stimulus
            node_ids = list(network.nodes.keys())
            selected_node = st.selectbox("Select Node for Stimulus", node_ids)
            
            col1, col2 = st.columns(2)
            
            with col1:
                stimulus_type = st.radio(
                    "Stimulus Type",
                    ["Impulse", "Continuous", "Rhythmic"]
                )
                
            with col2:
                stimulus_strength = st.slider(
                    "Stimulus Strength",
                    -5.0, 5.0, 1.0
                )
                
            if st.button("Apply Stimulus"):
                # Create stimulus pattern
                if stimulus_type == "Impulse":
                    external_input = {selected_node: stimulus_strength}
                    state = self.run_simulation_step(network, external_input=external_input)
                    st.success(f"Applied impulse to {selected_node}")
                    
                elif stimulus_type == "Continuous":
                    with st.spinner("Applying continuous stimulus..."):
                        for _ in range(50):
                            external_input = {selected_node: stimulus_strength}
                            state = self.run_simulation_step(network, external_input=external_input)
                            
                elif stimulus_type == "Rhythmic":
                    with st.spinner("Applying rhythmic stimulus..."):
                        for i in range(100):
                            if i % 10 < 5:  # On for 5 steps, off for 5 steps
                                external_input = {selected_node: stimulus_strength}
                            else:
                                external_input = None
                            state = self.run_simulation_step(network, external_input=external_input)
                            
                # Generate and play resulting audio
                audio = self.generate_audio(network, duration=1.0)
                audio_html = self.create_audio_player(audio)
                st.markdown("### Network Response")
                st.markdown(audio_html, unsafe_allow_html=True)
                
            # Real-time parameter adjustment
            st.subheader("Real-time Parameter Control")
            
            with st.expander("Adjust Network Parameters"):
                new_coupling = st.slider(
                    "Coupling Strength",
                    0.0, 10.0, 
                    network.coupling_strength,
                    key="realtime_coupling"
                )
                
                new_damping = st.slider(
                    "Global Damping",
                    0.0, 1.0,
                    network.global_damping,
                    key="realtime_damping"
                )
                
                if st.button("Update Parameters"):
                    network.coupling_strength = new_coupling
                    network.global_damping = new_damping
                    st.success("Parameters updated!")
                    
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Built with ❤️ using SynthNN - Synthetic Resonant Neural Networks</p>
            <p>Explore the intersection of neuroscience, physics, and music</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    demo = SynthNNDemo()
    demo.main() 