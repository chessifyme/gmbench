import re

_FENCE = re.compile(r"^\s*```.*$")
_HTML = re.compile(r"<[^>]+>")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"^\s*[-*+]\s+")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![A-Za-z0-9])([*_])(?=\S)(.+?)(?<=\S)\1(?![A-Za-z0-9])")


def to_plain_text(*, markdown: str) -> str:
    lines: list[str] = []
    for raw in markdown.splitlines():
        if _FENCE.match(raw):
            continue
        line = _HTML.sub("", raw)
        line = _IMAGE.sub(r"\1", line)
        line = _LINK.sub(r"\1", line)
        line = _INLINE_CODE.sub(r"\1", line)
        line = _HEADING.sub("", line)
        line = _BULLET.sub("", line)
        line = _BOLD.sub(r"\2", line)
        line = _ITALIC.sub(r"\2", line)
        lines.append(line.rstrip())
    return "\n".join(lines).strip()
