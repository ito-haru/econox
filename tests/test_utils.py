"""
Tests for general utility functions in econox.utils.
Covers get_from_pytree and set_in_pytree.
"""

import pytest
import equinox as eqx
from collections import namedtuple

from econox.utils import get_from_pytree, set_in_pytree


# =============================================================================
# set_in_pytree
# =============================================================================

class TestSetInPytree:

    # --- dict ---

    def test_set_existing_key_in_dict(self):
        d = {"a": 1, "b": 2}
        result = set_in_pytree(d, "a", 99)
        assert result["a"] == 99
        assert result["b"] == 2

    def test_set_new_key_in_dict(self):
        d = {"a": 1}
        result = set_in_pytree(d, "new_key", 42)
        assert result["new_key"] == 42

    def test_set_dict_does_not_mutate_original(self):
        d = {"a": 1}
        set_in_pytree(d, "a", 99)
        assert d["a"] == 1

    # --- eqx.Module (attribute access) ---

    def test_set_attr_on_eqx_module(self):
        class MyModule(eqx.Module):
            x: float
            y: float

        m = MyModule(x=1.0, y=2.0)
        result = set_in_pytree(m, "x", 9.0)
        assert result.x == 9.0
        assert result.y == 2.0

    def test_set_attr_does_not_mutate_original_module(self):
        class MyModule(eqx.Module):
            x: float

        m = MyModule(x=1.0)
        set_in_pytree(m, "x", 9.0)
        assert m.x == 1.0

    # --- error ---

    def test_raises_on_missing_attr(self):
        class SimpleObj(eqx.Module):
            x: float

        obj = SimpleObj(x=1.0)
        with pytest.raises(AttributeError):
            set_in_pytree(obj, "nonexistent", 0)


# =============================================================================
# get_from_pytree (regression guard — was present before this branch)
# =============================================================================

class TestGetFromPytree:

    def test_get_from_dict(self):
        d = {"a": 10}
        assert get_from_pytree(d, "a") == 10

    def test_get_from_dict_missing_with_default(self):
        d = {"a": 10}
        assert get_from_pytree(d, "b", default=99) == 99

    def test_get_from_dict_missing_raises(self):
        with pytest.raises(KeyError):
            get_from_pytree({}, "missing")

    def test_get_from_attribute(self):
        class Obj(eqx.Module):
            val: float

        obj = Obj(val=3.14)
        assert get_from_pytree(obj, "val") == 3.14

    def test_get_from_attribute_missing_raises(self):
        class Obj(eqx.Module):
            val: float

        with pytest.raises(AttributeError):
            get_from_pytree(Obj(val=1.0), "nonexistent")
