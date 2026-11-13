"""Print every reviewer's personal access link (make review_links, for the local app): one line per name in REVIEW_ANNOTATORS
and REVIEW_ADJUDICATORS. Send each reviewer their own link only; it signs them in (review_app.identity). --revoke NAME makes
that reviewer's old link stop working at once and prints their new one."""

import argparse

from bench.config import ReviewEnv
from bench.paths import EXPORTS_DIR
from review_app.identity import GENERATIONS_FILE_NAME, Reviewers, load_access_secret, revoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="the review app's address, e.g. https://review.example.org")
    parser.add_argument("--revoke", default=None, help="a reviewer's name: their old link stops working, and only their new link is printed")
    args = parser.parse_args()
    env = ReviewEnv()
    data_dir = env.review_data_dir or EXPORTS_DIR
    try:
        reviewers = Reviewers(annotators=env.annotator_names(), adjudicators=env.adjudicator_names(), secret=load_access_secret(data_dir=data_dir), generations_file=data_dir / GENERATIONS_FILE_NAME)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.revoke is not None:
        if args.revoke not in reviewers.names:
            raise SystemExit(f"{args.revoke!r} is not in REVIEW_ANNOTATORS or REVIEW_ADJUDICATORS")
        revoke(path=data_dir / GENERATIONS_FILE_NAME, name=args.revoke)
    for name, link in reviewers.links(base_url=args.base_url).items():
        if args.revoke is not None and name != args.revoke:
            continue
        role = "adjudicator" if name in reviewers.adjudicators else "reviewer"
        print(f"{name} ({role}): {link}")


if __name__ == "__main__":
    main()
