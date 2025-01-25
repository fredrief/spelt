# pylint: disable=protected-access

from enum import Enum, auto
from typing import Optional, Dict, List, Union, Tuple, Any, TypeVar, overload, Callable, Set
import numpy as np
from pint import UnitRegistry
import json
from pathlib import Path
from dataclasses import dataclass

ureg = UnitRegistry()

# Add type variable for numeric types
NumericType = TypeVar('NumericType', int, float, np.number)

class SignalDomain(Enum):
    UNDEFINED = "undefined"
    TIME = "time"
    FREQUENCY = "frequency"

class SamplingType(Enum):
    UNDEFINED = "undefined"
    REGULAR = "regular"
    IRREGULAR = "irregular"

class MetadataKeys(Enum):
    """Enumeration of standard metadata keys."""
    SAMPLING_TYPE = "sampling_type"
    UNIT = "unit"
    NAME = "name"
    DESCRIPTION = "description"
    X_UNIT = "x_unit"
    X_NAME = "x_name"
    X_INTERVAL = "x_interval"
    DIMENSIONS = "dimensions"
    DIM_UNITS = "dim_units"
    DOMAIN = "domain"

@dataclass
class MetadataValidator:
    """Validation rules for metadata."""
    required: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    error_msg: str = "Invalid value"

# Add after existing enums, before Signal class
class MetadataValidation:
    """Metadata validation rules and utilities."""

    @staticmethod
    def is_valid_unit(value: str) -> bool:
        """Validate unit string."""
        try:
            ureg.parse_expression(value)
            return True
        except:
            return False

    @staticmethod
    def is_valid_dimension_list(value: List[str]) -> bool:
        """Validate dimension names."""
        return isinstance(value, list) and all(isinstance(x, str) for x in value)

    @staticmethod
    def is_positive_number(value: Union[int, float]) -> bool:
        """Validate positive numeric value."""
        return isinstance(value, (int, float)) and value > 0

    @staticmethod
    def is_valid_sampling_type(value: Any) -> bool:
        """Validate sampling type."""
        if isinstance(value, SamplingType):
            return True
        if isinstance(value, str):
            try:
                SamplingType(value)
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def is_valid_domain(value: Any) -> bool:
        """Validate domain."""
        if isinstance(value, SignalDomain):
            return True
        if isinstance(value, str):
            try:
                SignalDomain(value)
                return True
            except ValueError:
                return False
        return False

    # Validation rules for each metadata key
    RULES = {
        MetadataKeys.SAMPLING_TYPE: MetadataValidator(
            required=True,
            validator=is_valid_sampling_type,
            error_msg="Sampling type must be a valid SamplingType value"
        ),
        MetadataKeys.UNIT: MetadataValidator(
            validator=is_valid_unit,
            error_msg="Invalid unit specification"
        ),
        MetadataKeys.NAME: MetadataValidator(
            validator=lambda x: isinstance(x, str),
            error_msg="Name must be a string"
        ),
        MetadataKeys.X_UNIT: MetadataValidator(
            validator=is_valid_unit,
            error_msg="Invalid x-axis unit specification"
        ),
        MetadataKeys.X_INTERVAL: MetadataValidator(
            validator=is_positive_number,
            error_msg="X interval must be a positive number"
        ),
        MetadataKeys.DIMENSIONS: MetadataValidator(
            required=True,
            validator=is_valid_dimension_list,
            error_msg="Dimensions must be a list of strings"
        ),
        MetadataKeys.DIM_UNITS: MetadataValidator(
            validator=lambda x: isinstance(x, list) and all(MetadataValidation.is_valid_unit(u) for u in x),
            error_msg="Dimension units must be a list of valid unit specifications"
        ),
        MetadataKeys.DOMAIN: MetadataValidator(
            required=True,
            validator=is_valid_domain,
            error_msg="Domain must be a valid SignalDomain value"
        )
    }

class Signal:
    """A class representing a signal with n-dimensional data and metadata.

    The Signal class is organized into the following sections:
    1. Core Methods - Basic initialization and representation
    2. Properties - Data and metadata access
    3. Data Access & Manipulation - Methods for data operations
    4. Arithmetic Operations - Mathematical operations between signals
    5. NumPy Integration - NumPy compatibility methods
    6. Metadata Operations - Methods for handling metadata
    """

    ###################
    # 1. Core Methods #
    ###################

    def __init__(
        self,
        data: Union[np.ndarray, List[NumericType]],
        metadata: Optional[Dict[str, Any]] = None,
        x_data: Optional[Union[np.ndarray, List[NumericType]]] = None
    ):
        """Initialize Signal with n-dimensional data."""
        try:
            self._data = np.asarray(data)
        except Exception as e:
            raise TypeError(f"Data must be convertible to numpy array: {str(e)}")

        if 0 in self._data.shape:
            raise ValueError(f"Data cannot have empty dimensions. Shape: {self._data.shape}")

        # Initialize default metadata
        self._metadata = {
            MetadataKeys.DOMAIN.value: SignalDomain.UNDEFINED.value,
            MetadataKeys.SAMPLING_TYPE.value: SamplingType.UNDEFINED.value,
            MetadataKeys.DIMENSIONS.value: self._infer_dimensions(),
        }

        # Validate and set x_data if provided
        if x_data is not None:
            try:
                self._x_data = np.asarray(x_data)
            except Exception as e:
                raise TypeError(f"x_data must be convertible to numpy array")

            # Validate dimensions
            if self._x_data.ndim != 1:
                raise ValueError(f"x_data must be one-dimensional, got shape {self._x_data.shape}")
            if self._x_data.shape[0] != self._data.shape[0]:
                raise ValueError(f"x_data length {self._x_data.shape[0]} must match data first dimension {self._data.shape[0]}")

            # Determine sampling type
            if self._x_data.size > 1:
                intervals = np.diff(self._x_data)
                if np.allclose(intervals, intervals[0]):
                    self._metadata[MetadataKeys.SAMPLING_TYPE.value] = SamplingType.REGULAR.value
                    self._metadata[MetadataKeys.X_INTERVAL.value] = float(intervals[0])
                else:
                    self._metadata[MetadataKeys.SAMPLING_TYPE.value] = SamplingType.IRREGULAR.value
        else:
            self._x_data = None

        # Update with provided metadata
        if metadata:
            self.update_metadata(metadata)

    def __repr__(self) -> str:
        """Return string representation."""
        domain = self._metadata[MetadataKeys.DOMAIN]
        sampling = self._metadata[MetadataKeys.SAMPLING_TYPE]
        name = self._metadata.get(MetadataKeys.NAME, "unnamed")
        return f"Signal(name='{name}', length={len(self)}, domain={domain}, sampling={sampling})"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        parts = []
        name = self._metadata.get(MetadataKeys.NAME, "unnamed signal")
        parts.append(name)
        parts.append(f"length: {len(self)}")
        if MetadataKeys.UNIT in self._metadata:
            parts.append(f"unit: {self._metadata[MetadataKeys.UNIT]}")
        if self._metadata[MetadataKeys.DOMAIN] != SignalDomain.UNDEFINED.value:
            parts.append(f"domain: {self._metadata[MetadataKeys.DOMAIN]}")
        sampling = self._metadata[MetadataKeys.SAMPLING_TYPE]
        if sampling != SamplingType.UNDEFINED.value:
            parts.append(f"sampling: {sampling}")
            if sampling == SamplingType.REGULAR.value and MetadataKeys.X_INTERVAL in self._metadata:
                parts.append(f"interval: {self._metadata[MetadataKeys.X_INTERVAL]}")
        return " | ".join(parts)

    def __len__(self) -> int:
        """Return length of signal data."""
        return self._data.size

    ################
    # 2. Properties #
    ################

    @property
    def data(self) -> np.ndarray:
        """Get signal data."""
        return self._data.copy()

    @property
    def domain(self) -> SignalDomain:
        """Get signal domain as enum."""
        return SignalDomain(self._metadata[MetadataKeys.DOMAIN])

    @property
    def metadata(self) -> dict:
        """Get metadata dictionary."""
        return self._metadata.copy()

    @property
    def shape(self) -> tuple:
        """Get data shape."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self.shape)

    @property
    def dimensions(self) -> List[str]:
        """Get list of dimension names."""
        return self._metadata[MetadataKeys.DIMENSIONS.value]

    @property
    def x_data(self) -> Optional[np.ndarray]:
        """Get x-axis data."""
        return self._x_data

    #################################
    # 3. Data Access & Manipulation #
    #################################

    def get_slice(self, **kwargs) -> 'Signal':
        """Get a slice of the signal.

        Args:
            **kwargs: Dimension name and index pairs
                     e.g., channel=0, time=slice(None)

        Returns:
            New Signal object with sliced data

        Raises:
            ValueError: If dimension name is invalid
        """
        # Get current dimensions
        dimensions = self._metadata[MetadataKeys.DIMENSIONS.value]

        # Build slice tuple
        slice_tuple = []
        for dim in dimensions:
            if dim in kwargs:
                slice_tuple.append(kwargs[dim])
            else:
                slice_tuple.append(slice(None))

        # Create new data array
        new_data = self._data[tuple(slice_tuple)]

        # Update metadata for new dimensions
        new_metadata = self._metadata.copy()
        new_dims = []
        for i, (dim, sl) in enumerate(zip(dimensions, slice_tuple)):
            if isinstance(sl, slice):
                new_dims.append(dim)
            elif isinstance(sl, (int, np.integer)):
                continue
            else:
                new_dims.append(dim)
        new_metadata[MetadataKeys.DIMENSIONS.value] = new_dims

        # Handle x_data if present
        new_x_data = None
        if self._x_data is not None and 'time' in kwargs:
            new_x_data = self._x_data[kwargs['time']]

        return Signal(data=new_data, metadata=new_metadata, x_data=new_x_data)

    ###########################
    # 4. Arithmetic Operations #
    ###########################

    def __add__(self, other: Union['Signal', NumericType]) -> 'Signal':
        """Add two signals or signal and scalar."""
        if isinstance(other, Signal):
            if self.shape != other.shape:
                raise ValueError("Signal shapes must match")
            return Signal(self._data + other._data, metadata=self._metadata.copy())
        return Signal(self._data + other, metadata=self._metadata.copy())

    def __radd__(self, other: NumericType) -> 'Signal':
        """Reverse add operation."""
        return Signal(other + self._data, metadata=self._metadata.copy())

    def __sub__(self, other: Union['Signal', NumericType]) -> 'Signal':
        """Subtract signal with another signal or scalar."""
        if isinstance(other, Signal):
            if self.shape != other.shape:
                raise ValueError("Signal shapes must match")
            return Signal(self._data - other._data, metadata=self._metadata.copy())
        return Signal(self._data - other, metadata=self._metadata.copy())

    def __rsub__(self, other: NumericType) -> 'Signal':
        """Reverse subtract operation."""
        return Signal(other - self._data, metadata=self._metadata.copy())

    def __mul__(self, other: Union['Signal', NumericType]) -> 'Signal':
        """Multiply signal with another signal or scalar."""
        if isinstance(other, Signal):
            if self.shape != other.shape:
                raise ValueError("Signal shapes must match")
            return Signal(self._data * other._data, metadata=self._metadata.copy())
        return Signal(self._data * other, metadata=self._metadata.copy())

    def __rmul__(self, other: NumericType) -> 'Signal':
        """Reverse multiply operation."""
        return Signal(other * self._data, metadata=self._metadata.copy())

    def __truediv__(self, other: Union['Signal', NumericType]) -> 'Signal':
        """Divide signal by another signal or scalar."""
        if isinstance(other, Signal):
            if self.shape != other.shape:
                raise ValueError("Signal shapes must match")
            return Signal(self._data / other._data, metadata=self._metadata.copy())
        return Signal(self._data / other, metadata=self._metadata.copy())

    def __rtruediv__(self, other: NumericType) -> 'Signal':
        """Reverse divide operation."""
        return Signal(other / self._data, metadata=self._metadata.copy())

    ########################
    # 5. NumPy Integration #
    ########################

    def __array__(self) -> np.ndarray:
        """Convert Signal to numpy array for numpy operations."""
        return self._data

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any
    ) -> Optional['Signal']:
        """Handle NumPy universal functions (add, multiply, etc.)."""
        arrays = [(x._data if isinstance(x, Signal) else x) for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)
        if result is None:
            return None
        return Signal(result)

    def __array_function__(
        self,
        func: Callable,
        types: Tuple[type, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any]
    ) -> Union['Signal', Any]:
        """Handle NumPy functions (mean, std, etc.)."""
        arrays = [(x._data if isinstance(x, Signal) else x) for x in args]
        result = func(*arrays, **kwargs)
        if isinstance(result, np.ndarray):
            return Signal(result)
        return result

    ##########################
    # 6. Metadata Operations #
    ##########################

    def _validate_metadata(self, key: MetadataKeys, value: Any) -> None:
        """Validate metadata value against defined rules.

        Args:
            key: Metadata key to validate
            value: Value to validate

        Raises:
            ValueError: If validation fails
        """
        if key not in MetadataValidation.RULES:
            return

        rule = MetadataValidation.RULES[key]
        if value is None:
            if rule.required:
                raise ValueError(f"Metadata key '{key.value}' is required")
            return

        if rule.validator and not rule.validator(value):
            raise ValueError(f"Invalid value for '{key.value}': {rule.error_msg}")

    def _validate_metadata_consistency(self) -> None:
        """Validate consistency between different metadata fields.

        Raises:
            ValueError: If metadata is inconsistent
        """
        # Validate dimensions match data shape
        dims = self._metadata.get(MetadataKeys.DIMENSIONS.value)
        if dims and len(dims) != len(self.shape):
            raise ValueError("Number of dimensions doesn't match data shape")

        # Validate dimension units if present
        dim_units = self._metadata.get(MetadataKeys.DIM_UNITS.value)
        if dim_units and len(dim_units) != len(dims):
            raise ValueError("Number of dimension units doesn't match number of dimensions")

        # Validate sampling type consistency
        sampling_type = self._metadata.get(MetadataKeys.SAMPLING_TYPE.value)
        if sampling_type == SamplingType.REGULAR.value:
            if MetadataKeys.X_INTERVAL.value not in self._metadata:
                raise ValueError("Regular sampling requires X_INTERVAL metadata")

    def _set_metadata(self, key: str, value: Any) -> None:
        """Set metadata with validation."""
        try:
            metadata_key = MetadataKeys(key)
        except ValueError:
            # Allow custom metadata keys
            self._metadata[key] = value
            return

        # Convert string values to Enums if applicable
        if metadata_key == MetadataKeys.DOMAIN and isinstance(value, str):
            value = SignalDomain(value)
        elif metadata_key == MetadataKeys.SAMPLING_TYPE and isinstance(value, str):
            value = SamplingType(value)

        # Validate
        self._validate_metadata(metadata_key, value)
        self._metadata[key] = value
        self._validate_metadata_consistency()

    def set_metadata(self, key: str, value: Any) -> 'Signal':
        """Set metadata with validation."""
        self._set_metadata(key, value)
        return self

    def update_metadata(self, metadata: Dict[str, Any]) -> 'Signal':
        """Update multiple metadata values at once."""
        # Create temporary copy to ensure atomicity
        temp_metadata = self._metadata.copy()
        try:
            for key, value in metadata.items():
                self._set_metadata(key, value)
        except ValueError as e:
            self._metadata = temp_metadata
            raise ValueError(f"Metadata update failed: {str(e)}")
        return self

    def _infer_dimensions(self) -> List[str]:
        """Infer dimension names from data shape."""
        return [f'dim_{i}' for i in range(len(self._data.shape))]

    def save(self, path: Union[str, Path]) -> None:
        """Save signal to disk.

        Args:
            path: Path to directory where signal will be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Create .signal marker file
        (path / '.signal').touch()

        # Save data
        np.save(path / 'data.npy', self._data)

        # Save x_data if it exists
        if self._x_data is not None:
            np.save(path / 'x_data.npy', self._x_data)

        # Convert Enums to strings for JSON serialization
        metadata_to_save = {}
        for key, value in self._metadata.items():
            if isinstance(value, Enum):
                metadata_to_save[key] = value.value
            else:
                metadata_to_save[key] = value

        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata_to_save, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Signal':
        """Load signal from disk.

        Args:
            path: Path to signal directory

        Returns:
            Loaded Signal object

        Raises:
            FileNotFoundError: If directory or required files don't exist
            ValueError: If directory is not a valid signal directory
        """
        path = Path(path)

        if not path.exists():
            raise ValueError("Directory not found")

        if not (path / '.signal').exists():
            raise ValueError("Not a valid signal directory")

        data_path = path / 'data.npy'
        if not data_path.exists():
            raise FileNotFoundError("Data file not found")

        data = np.load(data_path)

        x_data = None
        x_data_path = path / 'x_data.npy'
        if x_data_path.exists():
            x_data = np.load(x_data_path)

        metadata = {}
        metadata_path = path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)

                # Convert strings back to Enums where applicable
                for key, value in loaded_metadata.items():
                    if key == MetadataKeys.DOMAIN.value:
                        metadata[key] = SignalDomain(value)
                    elif key == MetadataKeys.SAMPLING_TYPE.value:
                        metadata[key] = SamplingType(value)
                    else:
                        metadata[key] = value

        return cls(data=data, metadata=metadata, x_data=x_data)

