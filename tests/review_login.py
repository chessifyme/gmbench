"""Signing a test client in as a reviewer, the way a personal access link does (review_app.identity)."""

from fastapi.testclient import TestClient

from review_app.identity import ACCESS_COOKIE, Reviewers, access_token

TEST_ACCESS_SECRET = b"test-access-secret"
TEST_ANNOTATORS = frozenset({"GM Test", "Aram", "Bella", "Lilit", "GM A", "GM B", "GM Other", "GM Old", "GM Grader"})


def reviewers_for(*, adjudicators: frozenset[str] = frozenset({"GM Judge", "Lead"}), annotators: frozenset[str] = TEST_ANNOTATORS) -> Reviewers:
    return Reviewers(annotators=annotators, adjudicators=adjudicators, secret=TEST_ACCESS_SECRET)


TEST_REVIEWERS = reviewers_for()


def login(client: TestClient, name: str) -> TestClient:
    """The client signed in as name; a name outside the configured reviewers gets a token no reviewer has."""
    client.cookies.set(ACCESS_COOKIE, access_token(secret=TEST_ACCESS_SECRET, name=name))
    return client
