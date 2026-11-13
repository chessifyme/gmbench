"""Who is using the review app. The reviewers are a fixed list (REVIEW_ANNOTATORS, and REVIEW_ADJUDICATORS for the adjudicators),
and each has a personal access link, /login#<token>, whose token is an HMAC-SHA256 of their name (and their link's generation)
under the access secret. The token sits in the link's fragment, which browsers never send, so it stays out of access logs; the
login page posts it and gets a cookie holding it. Every request maps the cookie back to the name, so a name can be neither
mistyped nor claimed by someone else. Raising one name's generation (python -m review_app.links --revoke NAME) revokes that
link alone; changing the secret (deleting its file) revokes every link at once (docs/operations.md, Review server)."""

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from bench.claim_tokens import secret_file

ACCESS_COOKIE = "reviewer_access"
ACCESS_COOKIE_SECONDS = 60 * 60 * 24 * 90
ACCESS_SECRET_FILE_NAME = "reviewer_access_secret"
GENERATIONS_FILE_NAME = "reviewer_access_generations.json"


def access_token(*, secret: bytes, name: str, generation: int = 0) -> str:
    salt = f"\0{generation}" if generation else ""
    return hmac.new(secret, f"gmbench reviewer access\0{name}{salt}".encode(), hashlib.sha256).hexdigest()[:32]


def read_generations(*, path: Path | None) -> dict[str, int]:
    """Each reviewer's link generation (0 when not listed): a revoked link's name has a higher one."""
    if path is None or not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in loaded.items()):
        raise ValueError(f"{path} must map reviewer names to whole numbers")
    return loaded


def revoke(*, path: Path, name: str) -> int:
    """Raise one reviewer's link generation, so their old link signs nobody in; returns the new generation."""
    generations = read_generations(path=path)
    generations[name] = generations.get(name, 0) + 1
    path.write_text(json.dumps(generations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return generations[name]


@dataclass(frozen=True)
class Reviewers:
    """The configured reviewers. An adjudicator sees every grandmaster's record of a task, records the resolved key, and alone
    rejects a task, clears a flag or takes a rejection back."""

    annotators: frozenset[str]
    adjudicators: frozenset[str]
    secret: bytes
    generations_file: Path | None = None

    def __post_init__(self) -> None:
        if not self.annotators and not self.adjudicators:
            raise ValueError("no reviewers are configured: set REVIEW_ANNOTATORS (and REVIEW_ADJUDICATORS) to the comma-separated names")
        if not self.secret:
            raise ValueError("the reviewer access secret is empty, so every access link would be guessable")

    @property
    def names(self) -> frozenset[str]:
        return self.annotators | self.adjudicators

    def name_for(self, *, token: str) -> str | None:
        """The reviewer whose access token this is, or None."""
        if not token:
            return None
        generations = read_generations(path=self.generations_file)
        return next((name for name in sorted(self.names) if secrets.compare_digest(access_token(secret=self.secret, name=name, generation=generations.get(name, 0)), token)), None)

    def token(self, *, name: str) -> str:
        return access_token(secret=self.secret, name=name, generation=read_generations(path=self.generations_file).get(name, 0))

    def links(self, *, base_url: str) -> dict[str, str]:
        return {name: f"{base_url.rstrip('/')}/login#{self.token(name=name)}" for name in sorted(self.names)}


def load_access_secret(*, data_dir: Path) -> bytes:
    """The access secret in the data folder, made (random, owner-only) on first use; its own file, so revoking the links does not
    change the claim tokens."""
    return secret_file(path=data_dir / ACCESS_SECRET_FILE_NAME, what="reviewer access secret", instead="delete it to revoke every access link")
