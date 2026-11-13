"""Opaque names for model claims in what a reviewer can read: the page's form fields and data attributes, and the workbook's
row_key column. A claim id follows the pooling order (the engine's claims, then one block of claims per model's answer), so
it would give away which claims came from the same answer, and by its place which model. A model claim is named instead by
`<task_id>_m<16 hex>`, an HMAC-SHA256 of its task and claim id under a secret only the operator holds; the app and the
workbook import map the name back to the claim id. Engine and grandmasters' claims keep their ids: their source is shown.

The secret is CLAIM_TOKEN_SECRET (environment or .env) when set, else the contents of a secret file, written with a new random
secret on first use (docs/operations.md, Review server)."""

import hashlib
import hmac
import os
import secrets
from pathlib import Path

from bench.config import ClaimTokenEnv
from bench.models import ReferenceClaim, Sheet

SECRET_FILE_NAME = "claim_token_secret"


def load_token_secret(*, path: Path) -> bytes:
    """CLAIM_TOKEN_SECRET when set; else the secret in path, made (random, readable by its owner only) when the file is missing.
    An empty secret is refused: every token would be guessable."""
    configured = ClaimTokenEnv().claim_token_secret
    if configured is not None and configured.get_secret_value().strip():
        return configured.get_secret_value().strip().encode("utf-8")
    return secret_file(path=path, what="claim token secret", instead="set CLAIM_TOKEN_SECRET")


def secret_file(*, path: Path, what: str, instead: str) -> bytes:
    """The secret in path, made (random, readable by its owner only) when the file is missing; an empty one is refused."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_hex(32) + "\n")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} holds no {what}: delete it to make a new one, or {instead}")
    return text.encode("utf-8")


def public_id(*, claim: ReferenceClaim, task_id: str, secret: bytes, recorded: frozenset[str] = frozenset()) -> str:
    """The name a reviewer can read for a claim: an opaque token for a model claim and for a claim a grandmaster added in their
    own record (recorded; its number would tell how many claims other grandmasters added), the claim id for any other."""
    opaque = claim.source.type == "model" or claim.claim_id in recorded
    return token(task_id=task_id, claim_id=claim.claim_id, secret=secret) if opaque else claim.claim_id


def recorded_claims(*, sheet: Sheet | None) -> frozenset[str]:
    """The claims grandmasters added in their own records (bench.annotations)."""
    return frozenset(claim_id for record in (sheet.annotations.values() if sheet else []) for claim_id in record.added)


def token(*, task_id: str, claim_id: str, secret: bytes) -> str:
    digest = hmac.new(secret, f"{task_id}\0{claim_id}".encode(), hashlib.sha256).hexdigest()[:16]
    return f"{task_id}_m{digest}"


def claim_ids(*, sheet: Sheet | None, secret: bytes) -> dict[str, str]:
    """Each public name of the sheet's claims, and each claim id itself (a form or workbook from before the tokens), to the
    claim id."""
    claims = sheet.claims if sheet else []
    recorded = recorded_claims(sheet=sheet)
    return {**{c.claim_id: c.claim_id for c in claims}, **{public_id(claim=c, task_id=sheet.task_id, secret=secret, recorded=recorded): c.claim_id for c in claims if sheet}}
