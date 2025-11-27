# src/econox/utils.py
"""
General utility functions shared across the Econox package.
"""

from typing import Any, TypeVar, Union

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
    # 1. Try dict-style access first (more explicit for dicts)
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if default is not _MISSING:
            return default
        raise KeyError(f"Key '{key}' not found in data dictionary.")
    
    # 2. Try mapping protocol (__getitem__ + __contains__ or keys)
    if hasattr(data, "__getitem__"):
        try:
            # Check if key exists (use __contains__ if available)
            if hasattr(data, "__contains__"):
                if key in data:
                    return data[key]
            else:
                # Fallback: try direct access
                return data[key]
        except (KeyError, TypeError, IndexError):
            # Continue to attribute access
            pass
    
    # 3. Try attribute access (for NamedTuple, dataclass, etc.)
    if hasattr(data, key):
        return getattr(data, key)
    
    # 4. Return default or raise error
    if default is not _MISSING:
        return default
        
    raise AttributeError(
        f"Could not find '{key}' in data object of type {type(data).__name__}. "
        "Ensure the data container has this key or attribute."
    )