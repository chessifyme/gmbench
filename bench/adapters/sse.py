import json
from collections.abc import Iterator

DONE_MARKER = "[DONE]"


def _event(*, data: list[str]) -> dict[str, object] | None:
    text = "\n".join(data)
    if text == DONE_MARKER:
        return None
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else None


def iter_sse(*, lines: Iterator[str]) -> Iterator[dict[str, object]]:
    """The JSON objects of a server-sent event stream; comment lines and the closing `[DONE]` marker carry no event."""
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                event = _event(data=data)
                data = []
                if event is not None:
                    yield event
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        event = _event(data=data)
        if event is not None:
            yield event
