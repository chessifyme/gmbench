from fastapi.testclient import TestClient

from tests.review_login import login


def sign_in_adjudicator(*, client: TestClient, name: str) -> None:
    """The client signed in as the adjudicator name, through their own access link (an adjudicator needs no other token)."""
    login(client, name)
