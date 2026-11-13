import ipaddress
import socket
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

import httpx

from bench.adapters.base import ALL_FAMILIES, Adapter, AgentCard, AnswerRecord, AnswerStatus
from bench.adapters.http_post import post_json
from bench.adapters.openai_compatible import PROBE_MESSAGE, PROBE_TIMEOUT_S
from bench.adapters.prompts import NEUTRAL_SYSTEM_PROMPT, render_user_message
from bench.config import entrant_env_value
from bench.entrants import EntrantCard, EntrantFileError, HttpEntrantFile, extract, fill, reply_json, resolve_headers
from bench.llm import LLMTransportError
from bench.models import Task

TRANSPORT_NAME = "http_entrant"


def env_lookup(variable: str) -> str | None:
    value = entrant_env_value(variable=variable)
    return value.get_secret_value() if value is not None else None


def entrant_card(*, entrant: EntrantCard) -> AgentCard:
    return AgentCard(
        name=entrant.name,
        version=entrant.version,
        provider=entrant.provider,
        declared_family=entrant.declared_family,
        access_method="api",
        attempted_families=ALL_FAMILIES,
        tools=frozenset(entrant.tools),
        affiliated_with_maintainer=entrant.affiliated_with_maintainer,
        label=entrant.label,
    )


class BlockedAddress(ValueError):
    """An endpoint a maintainers' re-run will not post to: not https, or a name that resolves to an address that is not public."""


def public_address(*, host: str, port: int) -> str:
    """The address to connect to for host: every address it resolves to must be public (ipaddress is_global), so a name that points
    at this machine or a private network (127.0.0.1.nip.io, 127.1, 10.x) is refused. A socket.gaierror is left to the caller."""
    addresses = sorted({str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
    private = [a for a in addresses if not ipaddress.ip_address(a.split("%")[0]).is_global]
    if not addresses or private:
        raise BlockedAddress(f"{host} resolves to an address that is not public")
    return addresses[0]


def pinned_request(*, url: str) -> tuple[str, dict[str, str], dict[str, object]]:
    """The url with its host replaced by the public address it resolves to now, the Host header and the TLS server name of the
    original host: the request goes to the address that was checked (no second lookup, so no DNS rebinding) and the certificate
    is still checked against the original name."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise BlockedAddress("a maintainers' re-run posts only to an https:// endpoint")
    port = parts.port or 443
    address = public_address(host=parts.hostname, port=port)
    shown = f"[{address}]" if ":" in address else address
    netloc = f"{shown}:{parts.port}" if parts.port else shown
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)), {"Host": parts.netloc}, {"sni_hostname": parts.hostname}


class HttpEntrantAdapter(Adapter):
    """An entrant described by entrants/<name>.yaml: one request per answer built from its body template, the answer read at its
    path in the JSON reply. The same contract as the OpenAI-compatible adapter (bench.adapters.http_post): a `transport:` error is
    retried once by the runner, a timeout never; the reply is recorded (raw.reply) so a submission can be checked, a header never."""

    def __init__(
        self,
        *,
        entrant: HttpEntrantFile,
        answer_timeout_s: float,
        lookup: Callable[[str], str | None] = env_lookup,
        transport: httpx.BaseTransport | None = None,
        public_only: bool = False,
    ) -> None:
        """public_only: a maintainers' re-run of a submission, which posts only over https to a public address, pinned per request."""
        if entrant.http.timeout_s is not None and entrant.http.timeout_s > answer_timeout_s:
            raise EntrantFileError(f"{entrant.name}: http.timeout_s {entrant.http.timeout_s:g} is over the benchmark's answer_timeout_s {answer_timeout_s:g}")
        self.name = entrant.name
        self.entrant = entrant
        self.declared_family = entrant.declared_family
        self.attempted_families = ALL_FAMILIES
        self.tools = frozenset(entrant.tools)
        self.affiliated_with_maintainer = entrant.affiliated_with_maintainer
        self._headers = resolve_headers(headers=entrant.http.headers, lookup=lookup)
        self._transport = transport
        self._public_only = public_only

    def card(self) -> AgentCard:
        return entrant_card(entrant=self.entrant)

    def _ask(self, *, message: str, timeout_s: float) -> tuple[AnswerStatus, str | None, str, dict[str, object]]:
        http = self.entrant.http
        limit = min(timeout_s, http.timeout_s) if http.timeout_s is not None else timeout_s
        body = fill(template=http.body, values={"message": message, "system": NEUTRAL_SYSTEM_PROMPT, "fresh_id": str(uuid.uuid4())})
        url, headers, extensions = http.url, self._headers, None
        if self._public_only:
            try:
                url, host, extensions = pinned_request(url=http.url)
            except BlockedAddress as error:
                return "error", f"blocked_address: {error}", "", {}
            except OSError as failure:
                return "error", f"transport: {type(failure).__name__}", "", {}
            headers = {**self._headers, **host}
        outcome = post_json(url=url, method=http.method, headers=headers, body=body, timeout_s=limit, transport=self._transport, extensions=extensions)
        if outcome.status != "ok":
            return outcome.status, outcome.error, "", {}
        reply, error = reply_json(content=outcome.content)
        if error is not None:
            return "error", error, "", {}
        status, error, text = extract(reply=reply, path=http.answer)
        return status, error, text, {"reply": reply}

    def check_available(self) -> None:
        status, error, _, _ = self._ask(message=PROBE_MESSAGE, timeout_s=PROBE_TIMEOUT_S)
        if status != "ok":
            raise LLMTransportError(f"{self.name} answered the availability probe with {error}")

    def answer(self, *, task: Task, run_index: int, timeout_s: float) -> AnswerRecord:
        started_at = datetime.now(UTC)
        status, error, text, raw = self._ask(message=render_user_message(task=task), timeout_s=timeout_s)
        finished_at = datetime.now(UTC)
        return AnswerRecord(
            task_id=task.task_id,
            entrant=self.name,
            run_index=run_index,
            text=text if status == "ok" else "",
            status=status,
            error=error,
            started_at=started_at,
            first_token_at=None,
            finished_at=finished_at,
            raw={"transport": TRANSPORT_NAME, "answer_path": self.entrant.http.answer, **raw, "streamed": False, "timing_measured": True},
        )
