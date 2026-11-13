"""One grandmaster edits a task at a time: the page holds a lock while it is open and renews it, and every write checks it."""

from dataclasses import dataclass, field

LOCK_SECONDS = 120.0
RENEW_SECONDS = 30


@dataclass
class EditLocks:
    seconds: float
    _held: dict[str, tuple[str, float]] = field(default_factory=dict)

    def holder(self, *, task_id: str, now: float) -> str | None:
        """Who holds the task now, or None when nobody does or the last renewal is too old."""
        held = self._held.get(task_id)
        if held is None or now - held[1] > self.seconds:
            return None
        return held[0]

    def hold(self, *, task_id: str, name: str, now: float) -> str | None:
        """Take or renew the lock for name; return the other holder's name instead when someone else holds it."""
        other = self.holder(task_id=task_id, now=now)
        if other is not None and other != name:
            return other
        self._held[task_id] = (name, now)
        return None

    def release(self, *, task_id: str, name: str) -> None:
        held = self._held.get(task_id)
        if held is not None and held[0] == name:
            del self._held[task_id]
