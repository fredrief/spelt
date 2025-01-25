import numpy as np
from pathlib import Path
from siva.signal import Signal

def generate_test_signals(output_dir: Path):
    """Generate various test signals."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Time vector
    t = np.linspace(0, 1, 1000)

    # Sine wave
    sine = Signal(
        data=np.sin(2 * np.pi * 10 * t),
        x_data=t,
        metadata={
            "name": "sine_wave",
            "unit": "V",
            "x_unit": "s",
            "description": "10 Hz sine wave"
        }
    )
    sine.save(output_dir / "sine_wave")

    # Noisy sine
    noisy_sine = Signal(
        data=np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t)),
        x_data=t,
        metadata={
            "name": "noisy_sine",
            "unit": "V",
            "x_unit": "s",
            "description": "10 Hz sine wave with noise"
        }
    )
    noisy_sine.save(output_dir / "noisy_sine")

    # Square wave
    square = Signal(
        data=np.sign(np.sin(2 * np.pi * 5 * t)),
        x_data=t,
        metadata={
            "name": "square_wave",
            "unit": "V",
            "x_unit": "s",
            "description": "5 Hz square wave"
        }
    )
    square.save(output_dir / "square_wave")

if __name__ == "__main__":
    # Create test data in a 'test_data' directory
    output_dir = Path(__file__).parent.parent / "test_data"
    generate_test_signals(output_dir)
    print(f"Test signals generated in {output_dir}")
