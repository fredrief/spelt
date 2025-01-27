#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from siva.signal import Signal, MetadataKeys, SignalDomain, SamplingType
import json

def generate_time_domain_signals(base_path: Path):
    """Generate time domain signals with different properties."""
    time_path = base_path / "time_domain"

    # Basic sine waves with different units
    t = np.linspace(0, 1, 1000)
    signals = [
        {
            "name": "voltage_sine",
            "data": 5 * np.sin(2 * np.pi * 10 * t),
            "x_data": t,
            "metadata": {
                MetadataKeys.NAME.value: "Voltage Sine Wave",
                MetadataKeys.UNIT.value: "V",
                MetadataKeys.X_UNIT.value: "s",
                MetadataKeys.X_NAME.value: "Time",
                MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": time_path / "voltage/sine_wave"
        },
        {
            "name": "current_sine",
            "data": 0.1 * np.sin(2 * np.pi * 10 * t),
            "x_data": t,
            "metadata": {
                MetadataKeys.NAME.value: "Current Sine Wave",
                MetadataKeys.UNIT.value: "A",
                MetadataKeys.X_UNIT.value: "s",
                MetadataKeys.X_NAME.value: "Time",
                MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": time_path / "current/sine_wave"
        },
        {
            "name": "temperature_sine",
            "data": 25 + 5 * np.sin(2 * np.pi * 0.1 * t),
            "x_data": t * 1000,  # milliseconds
            "metadata": {
                MetadataKeys.NAME.value: "Temperature Oscillation",
                MetadataKeys.UNIT.value: "°C",
                MetadataKeys.X_UNIT.value: "ms",
                MetadataKeys.X_NAME.value: "Time",
                MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": time_path / "temperature/oscillation"
        }
    ]

    # Non-uniform sampled signal
    t_nonuniform = np.sort(np.random.uniform(0, 1, 500))
    signals.append({
        "name": "pressure_nonuniform",
        "data": 1000 + 100 * np.sin(2 * np.pi * 5 * t_nonuniform),
        "x_data": t_nonuniform,
        "metadata": {
            MetadataKeys.NAME.value: "Pressure (Non-uniform)",
            MetadataKeys.UNIT.value: "Pa",
            MetadataKeys.X_UNIT.value: "s",
            MetadataKeys.X_NAME.value: "Time",
            MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
            MetadataKeys.SAMPLING_TYPE.value: SamplingType.IRREGULAR.value
        },
        "path": time_path / "pressure/nonuniform"
    })

    # Different length signals
    t_short = np.linspace(0, 0.1, 100)
    t_long = np.linspace(0, 10, 10000)
    signals.extend([
        {
            "name": "short_pulse",
            "data": np.exp(-(t_short - 0.05)**2 / 0.001),
            "x_data": t_short,
            "metadata": {
                MetadataKeys.NAME.value: "Short Pulse",
                MetadataKeys.UNIT.value: "V",
                MetadataKeys.X_UNIT.value: "s",
                MetadataKeys.X_NAME.value: "Time",
                MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": time_path / "voltage/short_pulse"
        },
        {
            "name": "long_decay",
            "data": np.exp(-0.2 * t_long),
            "x_data": t_long,
            "metadata": {
                MetadataKeys.NAME.value: "Long Decay",
                MetadataKeys.UNIT.value: "A",
                MetadataKeys.X_UNIT.value: "s",
                MetadataKeys.X_NAME.value: "Time",
                MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": time_path / "current/decay"
        }
    ])

    return signals

def generate_frequency_domain_signals(base_path: Path):
    """Generate frequency domain signals."""
    freq_path = base_path / "frequency_domain"

    # Linear frequency sweep
    f = np.linspace(0, 1000, 1001)
    signals = [
        {
            "name": "voltage_spectrum",
            "data": np.exp(-(f - 500)**2 / 10000),
            "x_data": f,
            "metadata": {
                MetadataKeys.NAME.value: "Voltage Spectrum",
                MetadataKeys.UNIT.value: "V",
                MetadataKeys.X_UNIT.value: "Hz",
                MetadataKeys.X_NAME.value: "Frequency",
                MetadataKeys.DOMAIN.value: SignalDomain.FREQUENCY.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": freq_path / "voltage/spectrum"
        },
        {
            "name": "impedance_sweep",
            "data": 50 + 10j * f/1000,  # Complex impedance
            "x_data": f,
            "metadata": {
                MetadataKeys.NAME.value: "Impedance Sweep",
                MetadataKeys.UNIT.value: "Ω",
                MetadataKeys.X_UNIT.value: "Hz",
                MetadataKeys.X_NAME.value: "Frequency",
                MetadataKeys.DOMAIN.value: SignalDomain.FREQUENCY.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": freq_path / "impedance/sweep"
        }
    ]

    # Log frequency sweep
    f_log = np.logspace(0, 6, 1001)  # 1 Hz to 1 MHz
    signals.append({
        "name": "bode_plot",
        "data": 20 * np.log10(1 / np.sqrt(1 + (f_log/1000)**2)),
        "x_data": f_log,
        "metadata": {
            MetadataKeys.NAME.value: "Bode Plot",
            MetadataKeys.UNIT.value: "dB",
            MetadataKeys.X_UNIT.value: "Hz",
            MetadataKeys.X_NAME.value: "Frequency",
            MetadataKeys.DOMAIN.value: SignalDomain.FREQUENCY.value,
            MetadataKeys.SAMPLING_TYPE.value: SamplingType.UNDEFINED.value
        },
        "path": freq_path / "transfer_function/bode"
    })

    return signals

def generate_spatial_domain_signals(base_path: Path):
    """Generate spatial domain signals."""
    space_path = base_path / "spatial_domain"

    # Linear space
    x = np.linspace(-10, 10, 1000)
    signals = [
        {
            "name": "gaussian_beam",
            "data": np.exp(-x**2 / 4),
            "x_data": x,
            "metadata": {
                MetadataKeys.NAME.value: "Gaussian Beam Profile",
                MetadataKeys.UNIT.value: "W/m²",
                MetadataKeys.X_UNIT.value: "mm",
                MetadataKeys.X_NAME.value: "Position",
                MetadataKeys.DOMAIN.value: SignalDomain.UNDEFINED.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": space_path / "optical/beam_profile"
        },
        {
            "name": "displacement",
            "data": 0.1 * np.sin(2 * np.pi * x / 5),
            "x_data": x * 1000,  # Convert to micrometers
            "metadata": {
                MetadataKeys.NAME.value: "Surface Displacement",
                MetadataKeys.UNIT.value: "nm",
                MetadataKeys.X_UNIT.value: "µm",
                MetadataKeys.X_NAME.value: "Position",
                MetadataKeys.DOMAIN.value: SignalDomain.UNDEFINED.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value
            },
            "path": space_path / "mechanical/displacement"
        }
    ]

    return signals

def generate_multidim_signals(base_path: Path):
    """Generate multi-dimensional test signals."""
    multidim_path = base_path / "multidim"

    # Time base for all signals
    t = np.linspace(0, 1, 1000)

    # 2D signal: Multiple voltage channels (L x M)
    # L = 1000 time points
    # M = 4 channels with different phases
    channels_2d = np.zeros((len(t), 4))
    for i in range(4):
        phase = i * np.pi / 4  # Different phase for each channel
        channels_2d[:, i] = 5 * np.sin(2 * np.pi * 10 * t + phase)

    signals = [
        {
            "name": "multichannel_voltage",
            "data": channels_2d,
            "x_data": t,
            "metadata": {
                MetadataKeys.NAME.value: "Multi-Channel Voltage",
                MetadataKeys.UNIT.value: "V",
                MetadataKeys.X_UNIT.value: "s",
                MetadataKeys.X_NAME.value: "Time",
                MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
                MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value,
                MetadataKeys.DIMENSIONS.value: ["time", "channel"]
            },
            "path": multidim_path / "voltage/multichannel"
        }
    ]

    # 3D signal: Multiple channels with different gains (L x M x N)
    # L = 1000 time points
    # M = 4 channels with different phases
    # N = 3 gain settings
    channels_3d = np.zeros((len(t), 4, 3))
    gains = [0.5, 1.0, 2.0]  # Different gain settings
    for i in range(4):
        phase = i * np.pi / 4
        base_signal = np.sin(2 * np.pi * 10 * t + phase)
        for j, gain in enumerate(gains):
            channels_3d[:, i, j] = 5 * gain * base_signal

    signals.append({
        "name": "voltage_gain_sweep",
        "data": channels_3d,
        "x_data": t,
        "metadata": {
            MetadataKeys.NAME.value: "Voltage Gain Sweep",
            MetadataKeys.UNIT.value: "V",
            MetadataKeys.X_UNIT.value: "s",
            MetadataKeys.X_NAME.value: "Time",
            MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
            MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value,
            MetadataKeys.DIMENSIONS.value: ["time", "channel", "gain"]
        },
        "path": multidim_path / "voltage/gain_sweep"
    })

    # 4D signal: Multiple channels, gains, and frequencies (L x M x N x P)
    # L = 1000 time points
    # M = 4 channels with different phases
    # N = 3 gain settings
    # P = 2 frequencies
    channels_4d = np.zeros((len(t), 4, 3, 2))
    freqs = [5, 10]  # Different frequencies
    for i in range(4):
        phase = i * np.pi / 4
        for j, gain in enumerate(gains):
            for k, freq in enumerate(freqs):
                channels_4d[:, i, j, k] = 5 * gain * np.sin(2 * np.pi * freq * t + phase)

    signals.append({
        "name": "voltage_freq_gain_sweep",
        "data": channels_4d,
        "x_data": t,
        "metadata": {
            MetadataKeys.NAME.value: "Voltage Frequency and Gain Sweep",
            MetadataKeys.UNIT.value: "V",
            MetadataKeys.X_UNIT.value: "s",
            MetadataKeys.X_NAME.value: "Time",
            MetadataKeys.DOMAIN.value: SignalDomain.TIME.value,
            MetadataKeys.SAMPLING_TYPE.value: SamplingType.REGULAR.value,
            MetadataKeys.DIMENSIONS.value: ["time", "channel", "gain", "frequency"]
        },
        "path": multidim_path / "voltage/freq_gain_sweep"
    })

    return signals

def create_signals_json(path: Path, signals: list):
    """Create .signals.json files in the given directory and all parent directories."""
    # Get all unique parent directories that should have .signals.json
    dirs_to_process = set()
    for signal in signals:
        signal_path = signal["path"]
        # Add all parent directories up to the base path
        current = signal_path.parent
        while str(current).startswith(str(path)):
            dirs_to_process.add(current)
            current = current.parent

    # Create .signals.json in each directory
    for directory in dirs_to_process:
        # Get all signals in this directory and its subdirectories
        dir_signals = []
        for signal in signals:
            if str(signal["path"]).startswith(str(directory)):
                dir_signals.append({
                    "name": signal["name"]
                })

        # Create .signals.json
        signals_file = directory / ".signals.json"
        with open(signals_file, 'w') as f:
            json.dump({"signals": dir_signals}, f, indent=4)
        print(f"Created {signals_file}")

def main():
    # Create base directory for test data
    base_path = Path("test_data")
    base_path.mkdir(exist_ok=True)

    # Generate all signals
    all_signals = []
    all_signals.extend(generate_time_domain_signals(base_path))
    all_signals.extend(generate_frequency_domain_signals(base_path))
    all_signals.extend(generate_spatial_domain_signals(base_path))
    all_signals.extend(generate_multidim_signals(base_path))  # Add multi-dimensional signals

    # Create directories and save signals
    for signal_info in all_signals:
        path = signal_info["path"]
        path.mkdir(parents=True, exist_ok=True)

        # Create signal object and save
        signal = Signal(
            data=signal_info["data"],
            metadata=signal_info["metadata"],
            x_data=signal_info.get("x_data")
        )
        signal.save(path)
        print(f"Created signal: {path}")

    # Create .signals.json files
    create_signals_json(base_path, all_signals)

if __name__ == "__main__":
    main()
