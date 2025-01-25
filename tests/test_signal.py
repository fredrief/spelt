import pytest
import numpy as np
from siva.signal import (
    Signal, SignalDomain, SamplingType, MetadataKeys,
    MetadataValidation, MetadataValidator
)
import tempfile
from pathlib import Path

##################
# Test Fixtures #
##################

@pytest.fixture
def basic_signal():
    """Fixture for a basic signal with just data."""
    return Signal(data=np.array([1, 2, 3, 4, 5]))

@pytest.fixture
def complex_signal():
    """Fixture for a signal with all parameters."""
    data = np.array([1, 2, 3, 4, 5])
    metadata = {
        MetadataKeys.DOMAIN.value: SignalDomain.TIME,
        MetadataKeys.SAMPLING_TYPE.value: SamplingType.UNDEFINED,
        MetadataKeys.UNIT.value: "V",
        MetadataKeys.NAME.value: "test_signal",
        MetadataKeys.X_UNIT.value: "s",
        MetadataKeys.X_NAME.value: "time",
        MetadataKeys.DIMENSIONS.value: ["dim_0"]
    }
    return Signal(data=data, metadata=metadata)

@pytest.fixture
def multidim_signal():
    """Fixture for multi-dimensional signal."""
    data = np.random.rand(10, 4, 3)  # time, channels, trials
    metadata = {
        MetadataKeys.DIMENSIONS.value: ['time', 'channels', 'trials'],
        MetadataKeys.DIM_UNITS.value: ['s', 'V', None],
        MetadataKeys.DOMAIN.value: SignalDomain.TIME.value
    }
    return Signal(data=data, metadata=metadata)

#######################
# Test Initialization #
#######################

class TestSignalInitialization:
    def test_basic_initialization(self, basic_signal):
        """Test initialization with minimal parameters."""
        assert len(basic_signal) == 5
        assert basic_signal.metadata[MetadataKeys.DOMAIN.value] == SignalDomain.UNDEFINED.value
        assert basic_signal.metadata[MetadataKeys.SAMPLING_TYPE.value] == SamplingType.UNDEFINED.value

    def test_dimension_inference(self):
        """Test automatic dimension name inference."""
        test_cases = [
            (np.zeros((4,)), ['dim_0']),
            (np.zeros((2, 2)), ['dim_0', 'dim_1']),
            (np.zeros((2, 3, 4)), ['dim_0', 'dim_1', 'dim_2'])
        ]
        for data, expected_dims in test_cases:
            signal = Signal(data)
            assert signal.dimensions == expected_dims

    def test_data_immutability(self, basic_signal):
        """Test that data is copied and immutable."""
        original_data = basic_signal.data
        original_data[0] = 100
        assert basic_signal.data[0] == 1

######################
# Test Data Handling #
######################

class TestDataValidation:
    def test_x_data_length_mismatch(self):
        """Test that x_data length must match data length."""
        with pytest.raises(ValueError):
            Signal(data=np.array([1, 2, 3]), x_data=np.array([1, 2, 3, 4]))

    def test_empty_data(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError):
            Signal(data=np.array([]))

#######################
# Test Metadata Logic #
#######################

class TestMetadataValidation:
    @pytest.mark.parametrize("key,valid_value,invalid_value", [
        (MetadataKeys.UNIT, "V", "invalid_unit"),
        (MetadataKeys.X_INTERVAL, 1.0, -1.0),
        (MetadataKeys.DIMENSIONS, ["time"], "not_a_list"),
    ])
    def test_metadata_validation(self, key, valid_value, invalid_value):
        """Test metadata validation for various keys."""
        # Test valid value
        signal = Signal(np.array([1, 2, 3]))
        signal.set_metadata(key.value, valid_value)

        # Test invalid value
        with pytest.raises(ValueError):
            signal.set_metadata(key.value, invalid_value)

    def test_metadata_consistency(self):
        """Test metadata consistency validation."""
        data = np.zeros((2, 3))
        with pytest.raises(ValueError, match="Number of dimensions"):
            Signal(data, {MetadataKeys.DIMENSIONS.value: ["time"]})  # Too few dimensions

    def test_custom_metadata(self):
        """Test setting custom metadata keys."""
        signal = Signal(np.array([1, 2, 3]))
        signal.set_metadata("custom_key", "custom_value")
        assert signal.metadata["custom_key"] == "custom_value"

    def test_atomic_metadata_update(self):
        """Test atomic metadata updates."""
        signal = Signal(np.array([1, 2, 3]))
        original_metadata = signal.metadata.copy()

        # Try invalid update
        with pytest.raises(ValueError):
            signal.update_metadata({
                MetadataKeys.UNIT.value: "V",
                MetadataKeys.X_INTERVAL.value: -1  # Invalid value
            })

        # Verify metadata unchanged
        assert signal.metadata == original_metadata

########################
# Test Signal Analysis #
########################

class TestSamplingTypeDetection:
    @pytest.mark.parametrize("x_data,expected_type", [
        (np.array([0, 1, 2, 3]), SamplingType.REGULAR),
        (np.array([0, 1, 2.5, 4]), SamplingType.IRREGULAR),
    ])
    def test_sampling_detection(self, x_data, expected_type):
        """Test automatic sampling type detection."""
        signal = Signal(np.array([1, 2, 3, 4]), x_data=x_data)
        assert signal.metadata[MetadataKeys.SAMPLING_TYPE.value] == expected_type.value

#########################
# Test Signal Operators #
#########################

class TestSignalArithmetic:
    @pytest.mark.parametrize("operation,expected", [
        (lambda s1, s2: s1 + s2, np.array([2, 3, 4, 5, 6])),
        (lambda s1, s2: s1 - s2, np.array([0, 1, 2, 3, 4])),
        (lambda s1, s2: s1 * s2, np.array([1, 2, 3, 4, 5])),
        (lambda s1, s2: s1 / s2, np.array([1, 2, 3, 4, 5])),
    ])
    def test_arithmetic_operations(self, basic_signal, operation, expected):
        """Test arithmetic operations between signals."""
        signal2 = Signal(data=np.ones_like(basic_signal.data))
        result = operation(basic_signal, signal2)
        assert isinstance(result, Signal)
        assert np.array_equal(result.data, expected)

class TestNumpyIntegration:
    @pytest.mark.parametrize("ufunc,expected", [
        (np.abs, np.array([1, 2, 3, 4, 5])),
        (np.sqrt, np.array([1, 1.4142, 1.7321, 2, 2.2361])),
    ])
    def test_numpy_ufuncs(self, basic_signal, ufunc, expected):
        """Test numpy universal functions."""
        result = ufunc(basic_signal)
        assert isinstance(result, Signal)
        assert np.allclose(result.data, expected, rtol=1e-4)

######################
# Test Signal I/O #
######################

class TestSignalIO:
    def test_save_load_roundtrip(self, tmp_path, complex_signal):
        """Test signal save and load roundtrip."""
        save_path = tmp_path / "test_signal"
        complex_signal.save(save_path)
        loaded = Signal.load(save_path)

        assert np.array_equal(loaded.data, complex_signal.data)
        assert loaded.metadata == complex_signal.metadata

    def test_save_load_basic(self):
        """Test saving and loading basic signal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.array([1, 2, 3, 4, 5])
            signal = Signal(data=data)
            signal.save(tmpdir + "/test_signal")

            loaded = Signal.load(tmpdir + "/test_signal")
            assert np.array_equal(loaded.data, data)
            assert loaded.x_data is None
            assert loaded.metadata[MetadataKeys.DOMAIN.value] == SignalDomain.UNDEFINED

    def test_save_load_complex(self, complex_signal):
        """Test saving and loading signal with all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            complex_signal.save(tmpdir + "/test_signal")

            loaded = Signal.load(tmpdir + "/test_signal")
            assert np.array_equal(loaded.data, complex_signal.data)
            assert np.array_equal(loaded.x_data, complex_signal.x_data)
            assert loaded.metadata[MetadataKeys.DOMAIN.value] == SignalDomain.TIME
            assert loaded.metadata[MetadataKeys.UNIT.value] == "V"

    def test_load_missing_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises(ValueError, match="Directory not found"):
            Signal.load("nonexistent_directory")

    def test_load_missing_data(self):
        """Test loading from directory without data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, '.signal').touch()
            with pytest.raises(FileNotFoundError, match="Data file not found"):
                Signal.load(tmpdir)

    def test_save_load_path_types(self):
        """Test saving and loading with different path types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.array([1, 2, 3])
            signal = Signal(data=data)

            # Test with string path
            signal.save(tmpdir + "/test_str")
            loaded_str = Signal.load(tmpdir + "/test_str")
            assert np.array_equal(loaded_str.data, data)

            # Test with Path object
            signal.save(Path(tmpdir) / "test_path")
            loaded_path = Signal.load(Path(tmpdir) / "test_path")
            assert np.array_equal(loaded_path.data, data)

    def test_save_load_metadata_types(self, tmp_path):
        """Test metadata types are preserved after save/load."""
        signal = Signal(np.array([1, 2, 3]))
        save_path = tmp_path / "test_signal"
        signal.save(save_path)
        loaded = Signal.load(save_path)
        assert isinstance(loaded.metadata[MetadataKeys.DOMAIN.value], SignalDomain)
        assert loaded.metadata[MetadataKeys.DOMAIN.value] == SignalDomain.UNDEFINED
        assert isinstance(loaded.metadata[MetadataKeys.SAMPLING_TYPE.value], SamplingType)
        assert loaded.metadata[MetadataKeys.SAMPLING_TYPE.value] == SamplingType.UNDEFINED

def test_signal_creation():
    # 1D signal
    data_1d = np.array([1, 2, 3, 4])
    signal = Signal(data_1d)
    assert signal.shape == (4,)
    assert signal.dimensions == ['dim_0']

    # 2D signal
    data_2d = np.array([[1, 2], [3, 4]])
    signal = Signal(data_2d)
    assert signal.shape == (2, 2)
    assert signal.dimensions == ['dim_0', 'dim_1']

    # Custom dimensions
    signal = Signal(data_2d, {MetadataKeys.DIMENSIONS.value: ['time', 'channels']})
    assert signal.dimensions == ['time', 'channels']

def test_signal_slicing():
    """Test signal slicing with default dimension names."""
    data = np.zeros((100, 4, 3))
    signal = Signal(data)
    sliced = signal.get_slice(dim_0=slice(None), dim_1=2)
    assert sliced.shape == (100, 3)

def test_signal_save_load(tmp_path):
    # Create test signal
    data = np.array([[1, 2], [3, 4]])
    metadata = {
        MetadataKeys.DIMENSIONS.value: ['dim_0', 'dim_1'],
        MetadataKeys.X_UNIT.value: 's',
        MetadataKeys.UNIT.value: 'V',
        MetadataKeys.DOMAIN.value: SignalDomain.TIME,
        MetadataKeys.SAMPLING_TYPE.value: SamplingType.UNDEFINED
    }
    signal = Signal(data, metadata)

    # Save and load
    save_path = tmp_path / 'test_signal'
    signal.save(save_path)
    loaded_signal = Signal.load(save_path)

    # Verify
    assert np.array_equal(loaded_signal._data, signal._data)
    assert loaded_signal._metadata == signal._metadata
    assert loaded_signal.dimensions == signal.dimensions

def test_invalid_dimensions():
    data = np.array([[1, 2], [3, 4]])

    # Test mismatched dimensions
    with pytest.raises(ValueError):
        Signal(data, {MetadataKeys.DIMENSIONS.value: ['time']})  # Too few dimensions

    with pytest.raises(ValueError):
        Signal(data, {MetadataKeys.DIMENSIONS.value: ['time', 'ch', 'extra']})  # Too many dimensions

def test_metadata_handling():
    data = np.array([1, 2, 3, 4])
    metadata = {
        MetadataKeys.NAME.value: 'test_signal',
        MetadataKeys.X_UNIT.value: 's',
        MetadataKeys.UNIT.value: 'V',
        MetadataKeys.DOMAIN.value: SignalDomain.TIME
    }
    signal = Signal(data, metadata)

    # Verify metadata is properly stored
    assert signal._metadata[MetadataKeys.NAME.value] == 'test_signal'
    assert signal._metadata[MetadataKeys.X_UNIT.value] == 's'
    assert signal._metadata[MetadataKeys.DOMAIN.value] == SignalDomain.TIME

def test_arithmetic_operations(basic_signal):
    """Test arithmetic operations without x_data expectations."""
    signal2 = Signal(data=np.array([1, 1, 1, 1, 1]))

    # Addition
    result = basic_signal + signal2
    assert isinstance(result, Signal)
    assert np.array_equal(result.data, basic_signal.data + signal2.data)

    # Other operations similarly...
