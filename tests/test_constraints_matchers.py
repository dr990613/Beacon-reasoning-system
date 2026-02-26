import pytest

from beacon_system.logic.matchers import match_spec


def test_match_spec_contains_and_not_contains():
    code = "def foo():\n    return 1\n"
    assert match_spec({"op": "contains", "value": "return 1"}, code)
    assert match_spec({"op": "not_contains", "value": "raise"}, code)


def test_match_spec_unknown_op():
    with pytest.raises(ValueError):
        match_spec({"op": "x"}, "code")
