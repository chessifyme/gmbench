"""The private-set origin the tests use: the working repository's own, or the stand-in tests/conftest.py installs without it."""

from bench.extensions import PRIVATE_ORIGINS

PRIVATE_ORIGIN = PRIVATE_ORIGINS[0]
