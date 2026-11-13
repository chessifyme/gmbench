"""Expected problems with what the user passed or set (a missing key, an unknown profile, a bad file): a command prints them on
one line, `error: <message>`, and exits 1, instead of a Python traceback."""

import sys
from collections.abc import Callable


class UserError(Exception):
    """A problem the user can fix from the message alone."""


def run_cli(main: Callable[[], None]) -> None:
    """Run a command's main(), printing a UserError on one line instead of a traceback."""
    try:
        main()
    except UserError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
