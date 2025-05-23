# Harmonic Resonant Network - AI-Driven Adaptive Retuning

## 🎉 Migration Complete: Now with GPU Acceleration!

The SynthNN music generation system has been successfully migrated to use the new `AcceleratedMusicalNetwork` framework. This brings:

- **GPU Acceleration**: Automatic detection and use of CUDA (NVIDIA) or Metal (Apple Silicon)
- **10-16x Performance Improvements**: Most operations now run faster than real-time
- **Full Backward Compatibility**: All existing code continues to work without modification
- **New Features**: Chord progressions, mode morphing, batch processing, and more

### Quick Start with Acceleration

```bash
# Run the accelerated music generation demo
python main.py --accelerated

# Run the migration test suite
python test_migration_complete.py

# Check the migration guide
cat MIGRATION_GUIDE.md
```

---

This project demonstrates the concept of **harmonic resonant networks** and their ability to **adapt to foreign signals** by recalculating a new harmonic structure in real-time. The nodes within this network represent resonating entities that start in harmonic tune, become dissonant upon the introduction of a foreign signal, and adapt by retuning themselves to match the new harmonic environment.

## Key Concepts

### 1. **Harmonically Tuned State**

In the initial state, the nodes are tuned to different **harmonic intervals** based on a base frequency. Harmonic intervals such as **unison (1:1)**, **perfect fifth (1.5:1)**, **fourth (1.33:1)**, and others are used to space out the nodes' frequencies. This creates a harmonious system where the nodes resonate in sync, but not in unison, reflecting musical intervals that naturally occur in physical systems.

### 2. **Dissonant State with Foreign Signal**

A **foreign signal** is introduced to the network, disrupting the harmonic state. This represents an external influence or disturbance that is not in tune with the existing harmonic structure. The nodes experience **dissonance**, as their frequencies no longer align with the new signal. This is visualized as chaotic or misaligned waveforms in the network's output.

### 3. **Adaptive Retuning**

To restore harmony, the network **analyzes the foreign signal**, estimates its dominant frequency using a Fourier transform, and calculates new harmonic intervals based on that frequency. Each node is **retuned** to fit within this new harmonic structure, creating a new resonant state that aligns with the foreign signal. This process mimics how musical instruments or resonating systems might naturally adapt to a changing environment or external force.

### 4. **Difference Between States**

The system also visualizes the **difference** between the initial harmonic state and the dissonant state, allowing us to measure how far the network has drifted due to the foreign signal. This gives insight into the magnitude of the dissonance and the scale of adaptation needed to retune the system.

## Visualizations

The network generates four key visualizations to demonstrate the process:

1. **Harmonically Tuned State**: The initial state where all nodes are in harmonic intervals relative to a base frequency.
2. **Foreign Signal with Dissonant State**: The dissonant state where the nodes are disrupted by the foreign signal, showing misaligned waveforms.
3. **Difference Between Harmonic and Dissonant States**: A visualization of the difference between the initial harmonic state and the dissonant state, highlighting how far the system has drifted.
4. **Retuned State in Harmony with Foreign Signal**: The final state after the nodes have been retuned to harmonize with the foreign signal, showing a new, aligned waveform.

## Usage

To run the network and visualize the adaptation process, simply execute the provided Python code. The network adapts dynamically to any foreign signal introduced and visualizes the harmonic, dissonant, and retuned states.

```bash
python main.py
```

## Future Directions

There are several possible extensions to this work:

- Dynamic Harmonic Structures: Explore other harmonic intervals (such as thirds, sixths, or sevenths) and allow the system to switch between them dynamically.
- Multimodal Harmonies: Introduce multiple foreign signals simultaneously and observe how the network adapts to find harmony across diverse frequencies.
- Custom Retuning Strategies: Experiment with various retuning strategies (e.g., minimizing energy or maximizing resonance) to see how different objectives impact the network's behavior.

## Extending the Network with Musical Modes:

Just like how modes in music use different scales built from the same notes but with different tonal centers, we can explore modal tuning for the nodes in the resonant network. Each mode could represent a unique harmonic structure the system can switch to when encountering different types of foreign signals.

For example:

- Major (Ionian): Nodes are tuned to harmonics that align with the major scale (e.g., unison, major third, perfect fifth).
- Dorian: A different modal structure could alter the harmonic intervals to reflect the Dorian scale.
- Lydian: The sharp fourth (tritone) interval could influence the system's retuning behavior.

### How This Could Work:

- Adaptive Modal Switching: The network could detect not only the dominant frequency of the foreign signal but also its harmonic content. Based on this, the system could switch modes (e.g., from Ionian to Dorian) to find the best harmonic fit.

- Modal Resonant Networks: Each mode would have a predefined set of harmonic intervals, and the network would retune the nodes according to the intervals of that specific mode. For instance, a Lydian mode might emphasize more dissonant intervals like tritones, while a Mixolydian mode might harmonize better with bluesy, dominant-seventh type signals.

Example of Modal Intervals:

- Ionian (Major): [1, 1.25, 1.5, 2, 2.5] (unison, major third, perfect fifth, octave, perfect fifth above octave)
- Dorian: [1, 1.22, 1.5, 2, 2.33] (with minor third and perfect fourth)
- Lydian: [1, 1.33, 1.67, 2, 2.5] (with augmented fourth or tritone)

### Implementation Idea:

We could introduce a mode-based harmonic retuning system. When the foreign signal is analyzed, the network could:

Select the mode whose intervals are best suited to the signal's harmonic content.
Retune the nodes according to the intervals of that mode, creating a new resonant structure that is both adaptive and reflective of the musical "mood."

This would not only increase the versatility of the system but also create a deeply musical framework for handling complex foreign data, akin to how musicians shift modes in improvisation to match the tonal landscape.

## Why Keep Records?

In AI-driven workflows, keeping records of the processes, ideas, and outcomes is crucial due to the way context and memory work in these systems. By documenting each step and the underlying concepts, we ensure that future iterations of the model can refer back to the reasoning behind current implementations, making it easier to transfer knowledge and extend the system in a meaningful way.

Contributing
Feel free to contribute ideas, bug fixes, or enhancements to the project. The adaptive retuning framework can serve as a basis for various research areas involving resonant systems, neural networks, or AI-driven harmony generation.

---

Author: AI-driven pair programming with enhancements by TheApeMachine
