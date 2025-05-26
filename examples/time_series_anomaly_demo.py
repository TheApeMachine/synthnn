"""Simple demo for time-series anomaly detection using SignalProcessor."""

import numpy as np
import matplotlib.pyplot as plt
from synthnn.core import SignalProcessor


def generate_signal(duration: float = 10.0, sample_rate: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic signal with injected anomalies."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    # Base signal: sine wave with noise
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
    
    # Inject anomalies at specific time points (20%, 60%, 85% of duration)
    anomaly_points = [int(0.2 * len(t)), int(0.6 * len(t)), int(0.85 * len(t))]
    for idx in anomaly_points:
        signal[idx] += 2.5 * np.random.choice([-1, 1])  # Add large positive or negative spike
    return t, signal


def main() -> None:
    """Run the anomaly detection demo."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
        return

    sp = SignalProcessor(sample_rate=100)
    t, sig = generate_signal()
    anomalies = sp.detect_anomalies(sig, window_size=50, threshold_factor=3.0)

    print(f"Generated signal with {len(sig)} samples")
    print(f"Detected {np.sum(anomalies)} anomaly points")

    plt.figure(figsize=(10, 3))
    plt.plot(t, sig, label="signal")
    plt.scatter(t[anomalies], sig[anomalies], color="red", label="anomaly")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Detected Anomalies")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
