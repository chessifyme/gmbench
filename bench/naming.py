import re

_UNDERSCORE_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def assert_underscore_name(*, name: str) -> None:
    if not _UNDERSCORE_NAME.fullmatch(name):
        raise ValueError(
            f"name {name!r} must contain only lowercase letters, digits and single underscores"
        )
