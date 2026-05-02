# src/econox/utils.py
"""
General utility functions shared across the Econox package.
"""

import copy
from typing import Any, TypeVar, Union
from collections.abc import Mapping
import equinox as eqx

# Sentinel value to distinguish "no default provided" from "default=None"
_MISSING = object()

T = TypeVar('T')

def get_from_pytree(
    data: Any, 
    key: str, 
    default: Union[T, object] = _MISSING
) -> Union[Any, T]:
    """
    Retrieve a value from a data container, supporting both dict-style (['key'])
    and attribute-style (.key) access.

    Args:
        data: The container (dict, NamedTuple, PyTree, etc.).
        key: The key or attribute name to retrieve.
        default: Value to return if key is not found. If not provided, raises error.

    Returns:
        The value associated with the key, or default if not found.

    Raises:
        KeyError: If data is dict-like and key is missing (and no default).
        AttributeError: If data is object-like and attribute is missing (and no default).
    """
    # 1. Try Mapping protocol (dict, etc.)
    # Use explicit check to avoid unintended sequence behavior
    if isinstance(data, Mapping):
        if key in data:
            return data[key]
        if default is not _MISSING:
            return default
        raise KeyError(f"Key '{key}' not found in data mapping of type {type(data).__name__}.")
    
    # 2. Try attribute access (NamedTuple, dataclass, class instance)
    if hasattr(data, key):
        return getattr(data, key)
    
    # 3. Try __getitem__ as a fallback (but be careful not to iterate)
    # This covers cases that have __getitem__ but are not Mappings (rare in this context but possible)
    if hasattr(data, "__getitem__"):
        try:
            return data[key]
        except (KeyError, TypeError, IndexError):
            pass
    
    # 4. Return default or raise error
    if default is not _MISSING:
        return default
        
    raise AttributeError(
        f"Could not find '{key}' in data object of type {type(data).__name__}. "
        f"Available keys/attributes could not be determined."
    )

def set_in_pytree(
    data: Any, 
    key: str, 
    value: Any
) -> Any:
    """
    Set a value in a data container, supporting both dict-style (['key'])
    and attribute-style (.key) access. Returns a new container with the updated value.

    Args:
        data: The container (dict, NamedTuple, PyTree, etc.).
        key: The key or attribute name to set.
        value: The new value to assign.

    Returns:
        A new container with the updated value.

    Raises:
        KeyError: If data is dict-like and key is missing.
        AttributeError: If data is object-like and attribute is missing.
    """
    # 1. Try Mapping protocol (dict, etc.)
    if isinstance(data, Mapping):
        new_data = dict(data)  # Create a shallow copy
        new_data[key] = value
        return new_data
    
    # 2. Try attribute access (NamedTuple, dataclass, class instance)
    if hasattr(data, key):
        return eqx.tree_at(lambda d: getattr(d, key), data, value)
    
    # 3. Try __setitem__ as a fallback (but be careful not to iterate)
    if hasattr(data, "__setitem__"):
        try:
            new_data = copy.copy(data)
            new_data[key] = value
            return new_data
        except (KeyError, TypeError, IndexError):
            pass
    
    raise AttributeError(
        f"Could not set '{key}' in data object of type {type(data).__name__}. "
        f"Available keys/attributes could not be determined."
    )
