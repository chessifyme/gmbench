"""Near-duplicate claims, found mechanically so a reviewer sees one idea's claims side by side and labels and weights it once.

No model is called and the result depends only on the texts: each claim becomes a set of features (its content words with
synonyms folded, say attacks/targets/eyes all to "pressure", plus the squares and moves it names). A feature that more than half
of a task's claims share (the judged move, say) tells them apart from nothing and is left out. Two claims are linked when their
features overlap by at least `LINK_JACCARD` (Jaccard) and they share at least one square or move; linked claims form a cluster
by single linkage. This is the rule of the claim-quality study: about two thirds of the linked pairs there were the same idea,
and almost all the rest overlapped heavily."""

import re

LINK_JACCARD = 0.30
COMMON_SHARE = 0.5
COMMON_MIN_CLAIMS = 4

SAN = re.compile(r"(?:O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x[a-h][1-8](?:=[QRBN])?|[a-h][1-8](?:=[QRBN])?)")
WORD = re.compile(r"O-O(?:-O)?|[A-Za-z][A-Za-z'\-]*[A-Za-z0-9]")
SQUARE = re.compile(r"[a-h][1-8]")
STOP = frozenset(
    """a an the and or of to in on at by for with from is are was were be been being it its this that these those as after before
    white black white's black's move moves position here there can could would should will may might has have had than then so but
    not no yes very more most less up down into over under also just only both each either which who what when where while his her
    their side piece pieces play plays playing player game good better best strong stronger weak weaker line keeps gives clear
    clearly slightly already still now immediately simply about against because since though around""".split()
)
SYNONYMS = {
    **dict.fromkeys(("pressures", "pressuring", "presses", "targets", "targeting", "target", "attacks", "attack", "attacking", "hits", "hitting", "bears", "eyes", "eyeing", "aims", "aiming"), "pressure"),
    **dict.fromkeys(("long-diagonal", "fianchetto", "fianchettoed"), "diagonal"),
    **dict.fromkeys(("develops", "development", "developing", "completes", "completing"), "develop"),
    **dict.fromkeys(("contests", "contesting", "challenge", "challenges", "challenging"), "contest"),
    **dict.fromkeys(("equalises", "equalizes", "equalise", "equalize", "equality", "balance", "balanced"), "equal"),
    **dict.fromkeys(("trades", "trading", "exchange", "exchanges", "exchanging", "simplify", "simplifying"), "trade"),
    **dict.fromkeys(("wins", "winning", "won"), "win"),
    **dict.fromkeys(("loses", "losing", "lost"), "lose"),
    **dict.fromkeys(("threatens", "threatening", "threats"), "threat"),
    **dict.fromkeys(("pins", "pinned", "pinning"), "pin"),
    **dict.fromkeys(("forks", "forking"), "fork"),
    **dict.fromkeys(("centre", "central"), "center"),
    **{plural: plural[:-1] for plural in ("pawns", "knights", "bishops", "rooks", "queens", "squares")},
}


def _is_move(token: str) -> bool:
    return token.startswith("O-O") or (SAN.fullmatch(token) is not None and (token[0] in "KQRBN" or len(token) <= 4))


def claim_features(*, text: str) -> frozenset[str]:
    """The claim's content words (lower case, synonyms folded, stop words out) and the squares and moves it names: a move also
    gives its destination square, so "Qb6" and "the queen on b6" share b6."""
    features: set[str] = set()
    for token in WORD.findall(text.replace("–", "-")):
        if _is_move(token):
            features.add(token)
            squares = SQUARE.findall(token)
            if squares:
                features.add(squares[-1])
            continue
        word = SYNONYMS.get(token.lower().strip("'"), token.lower().strip("'"))
        if word not in STOP and len(word) >= 3:
            features.add(word)
    return frozenset(features)


def _anchor(feature: str) -> bool:
    return feature.startswith("O-O") or SQUARE.search(feature) is not None


def near_duplicate_clusters(*, texts: dict[str, str]) -> list[list[str]]:
    """The clusters of two or more near-duplicate claims among texts (claim id to text), each in the order of texts, the
    clusters in the order of their first member. Deterministic: the same texts in the same order give the same clusters."""
    ids = list(texts)
    raw = {claim_id: claim_features(text=text) for claim_id, text in texts.items()}
    common: set[str] = set()
    if len(ids) >= COMMON_MIN_CLAIMS:
        counts: dict[str, int] = {}
        for features in raw.values():
            for feature in features:
                counts[feature] = counts.get(feature, 0) + 1
        common = {feature for feature, count in counts.items() if count > COMMON_SHARE * len(ids)}
    features = {claim_id: raw[claim_id] - common for claim_id in ids}
    parent = {claim_id: claim_id for claim_id in ids}

    def root(claim_id: str) -> str:
        while parent[claim_id] != claim_id:
            claim_id = parent[claim_id]
        return claim_id

    for i, first in enumerate(ids):
        for second in ids[i + 1 :]:
            a, b = features[first], features[second]
            shared = a & b
            if shared and any(_anchor(f) for f in shared) and len(shared) / len(a | b) >= LINK_JACCARD:
                parent[root(second)] = root(first)
    members: dict[str, list[str]] = {}
    for claim_id in ids:
        members.setdefault(root(claim_id), []).append(claim_id)
    return [group for group in members.values() if len(group) > 1]
