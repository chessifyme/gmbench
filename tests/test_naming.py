import pytest

from bench.naming import assert_underscore_name


def test_underscore_name_rejects_hyphen() -> None:
    with pytest.raises(ValueError):
        assert_underscore_name(name="pos-0001")


@pytest.mark.parametrize("name", ["pos 0001", "Pos_0001", "", "pos.0001"])
def test_underscore_name_rejects_space_uppercase_empty_and_dots(name: str) -> None:
    with pytest.raises(ValueError):
        assert_underscore_name(name=name)


@pytest.mark.parametrize("name", ["ap_0001", "gemini_default", "run_20261017_120000"])
def test_underscore_name_accepts_lowercase_digits_and_underscores(name: str) -> None:
    assert_underscore_name(name=name)
