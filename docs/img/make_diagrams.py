"""Draw the scale diagrams in docs/methodology.md and docs/grading_rules.md as SVG, a light and a dark file each, in the palette of pipeline-*.svg:
eval-bands-*.svg (the evaluation bands), move-labels-*.svg (the move labels by drop in win probability) and
scores-*.svg (coverage saturation and the precision ceiling). Every cutoff is read from config/bench.yaml.

Run from the repository root: uv run python docs/img/make_diagrams.py. tests/test_doc_images.py checks that the committed
files match what this script draws."""

from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent
CONFIG = OUT.parent.parent / "config" / "bench.yaml"
W = 900
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# the pipeline-*.svg palette, plus a blue ramp for the bands and a red for blunders
THEMES = {
    "light": {
        "bg": "#ffffff", "frame": "#d0d7de", "text": "#1f2328", "muted": "#59636e", "grid": "#d0d7de", "line": "#1f5a96",
        "neutral": ("#f6f8fa", "#8c959f", "#1f2328"),
        "bands": [("#e3ecf6", "#1f5a96", "#1f2328"), ("#c5d8ee", "#1f5a96", "#1f2328"), ("#9dbde2", "#1f5a96", "#1f2328"), ("#1f5a96", "#1f5a96", "#ffffff")],
        "labels": [("#e1eee6", "#22663f", "#22663f"), ("#f6f8fa", "#8c959f", "#1f2328"), ("#f6ecd9", "#7d500c", "#7d500c"), ("#f8dcc2", "#9a4a0c", "#7a3a08"), ("#ffebe9", "#a40e26", "#a40e26")],
    },
    "dark": {
        "bg": "#0d1117", "frame": "#30363d", "text": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d", "line": "#8ab8e8",
        "neutral": ("#151b23", "#656c76", "#e6edf3"),
        "bands": [("#16212e", "#8ab8e8", "#e6edf3"), ("#1d3550", "#8ab8e8", "#e6edf3"), ("#274b73", "#8ab8e8", "#e6edf3"), ("#8ab8e8", "#8ab8e8", "#0d1117")],
        "labels": [("#15241b", "#7fc49d", "#7fc49d"), ("#151b23", "#656c76", "#e6edf3"), ("#272013", "#e3b46a", "#e3b46a"), ("#2e1d0f", "#f0a35e", "#f0a35e"), ("#2d1416", "#ff8a80", "#ff8a80")],
    },
}


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, s: str, *, fill: str, size: float = 12.5, anchor: str = "middle", weight: str = "", mono: bool = False) -> str:
    extra = (f' font-weight="{weight}"' if weight else "") + (f' font-family="{MONO}"' if mono else "")
    return f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" fill="{fill}" text-anchor="{anchor}"{extra}>{esc(s)}</text>'


def heading(x: float, y: float, s: str, *, t: dict) -> str:
    return text(x, y, s, fill=t["muted"], size=11.5, anchor="start", weight="600", mono=True)


def svg(*, height: int, title: str, desc: str, body: list[str], t: dict) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" role="img" aria-labelledby="t d" font-family="{SANS}">\n'
        f'<title id="t">{esc(title)}</title>\n<desc id="d">{esc(desc)}</desc>\n'
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="8" fill="{t["bg"]}" stroke="{t["frame"]}"/>\n'
    )
    return head + "\n".join(body) + "\n</svg>\n"


def pawns(cp: int) -> str:
    return f"{cp / 100:.2f}"


def eval_bands(*, cfg: dict, t: dict) -> str:
    b = cfg["eval_bands"]
    edges = [b["slightly_better_cp"], b["better_cp"], b["large_edge_cp"], b["winning_cp"]]
    margin = b["margin_cp"]
    span = edges[-1] + 50  # the axis runs to half a pawn past the winning edge
    left, right = 40, W - 40
    y0, h = 58, 54

    def x(cp: float) -> float:
        return left + (cp + span) / (2 * span) * (right - left)

    names = [("slight", "edge"), ("clear", "edge"), ("large", "edge"), ("winning", "")]
    body = [heading(left, 30, "EVALUATION BANDS · pawns, from White's view; Black's bands mirror White's", t=t)]
    # equal band in the middle, then each side outwards
    fill, stroke, ink = t["neutral"]
    body.append(f'<rect x="{x(-edges[0]):g}" y="{y0}" width="{x(edges[0]) - x(-edges[0]):g}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    body.append(text(x(0), y0 + 25, "roughly", fill=ink, size=13, weight="700"))
    body.append(text(x(0), y0 + 41, "equal", fill=ink, size=13, weight="700"))
    bounds = edges + [span]
    for sign in (1, -1):
        for i, (top, bottom) in enumerate(names):
            lo, hi = bounds[i], bounds[i + 1]
            fill, stroke, ink = t["bands"][i]
            x1, x2 = sorted((x(sign * lo), x(sign * hi)))
            body.append(f'<rect x="{x1:g}" y="{y0}" width="{x2 - x1:g}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
            mid = (x1 + x2) / 2
            if bottom:
                body.append(text(mid, y0 + 25, top, fill=ink, size=13, weight="700"))
                body.append(text(mid, y0 + 41, bottom, fill=ink, size=13, weight="700"))
            else:
                body.append(text(mid, y0 + 33, top, fill=ink, size=13, weight="700"))
    # the open ends: winning goes on past the axis
    for sign in (1, -1):
        xe = x(sign * span)
        body.append(text(xe + sign * -6, y0 + h + 16, "…", fill=t["muted"], size=12.5, anchor="end" if sign > 0 else "start"))
    # ticks at every edge
    for cp in [-e for e in reversed(edges)] + [0] + edges:
        body.append(f'<line x1="{x(cp):g}" y1="{y0 + h}" x2="{x(cp):g}" y2="{y0 + h + 6}" stroke="{t["muted"]}"/>')
        label = "0" if cp == 0 else ("+" if cp > 0 else "−") + pawns(abs(cp))
        body.append(text(x(cp), y0 + h + 20, label, fill=t["muted"], size=11.5, mono=True))
    body.append(text(left, y0 - 8, "◀ Black better", fill=t["muted"], size=12, anchor="start"))
    body.append(text(right, y0 - 8, "White better ▶", fill=t["muted"], size=12, anchor="end"))
    # the margin, shown on the clear edge
    lo, hi = edges[1] - margin, edges[2] + margin
    ym = y0 + h + 46
    body.append(f'<line x1="{x(lo):g}" y1="{ym}" x2="{x(hi):g}" y2="{ym}" stroke="{t["line"]}" stroke-width="2"/>')
    for cp in (lo, hi):
        body.append(f'<line x1="{x(cp):g}" y1="{ym - 6}" x2="{x(cp):g}" y2="{ym + 6}" stroke="{t["line"]}" stroke-width="2"/>')
    for cp in (edges[1], edges[2]):
        body.append(f'<line x1="{x(cp):g}" y1="{y0 + h + 26}" x2="{x(cp):g}" y2="{ym}" stroke="{t["grid"]}" stroke-dasharray="3 3"/>')
    body.append(text(x(lo) - 10, ym + 4, f"“clearly better” holds from +{pawns(lo)} to +{pawns(hi)}: each band reaches {margin} cp past its edges", fill=t["text"], size=12.5, anchor="end"))
    body.append(text(left, ym + 34, "An edge belongs to the band above it: +0.70 is a clear edge.", fill=t["muted"], size=12, anchor="start"))
    desc = (
        "Evaluation bands in pawns from White's view, mirrored for Black: roughly equal below "
        f"{pawns(edges[0])}, slight edge from {pawns(edges[0])} to {pawns(edges[1])}, clear edge from {pawns(edges[1])} to {pawns(edges[2])}, "
        f"large edge from {pawns(edges[2])} to {pawns(edges[3])}, winning from {pawns(edges[3])}. An edge belongs to the band above it. "
        f"A band claim holds {margin} cp past its edges, so clearly better is accepted from {pawns(lo)} to {pawns(hi)}."
    )
    return svg(height=228, title="Evaluation bands", desc=desc, body=body, t=t)


def move_labels(*, cfg: dict, t: dict) -> str:
    lab = cfg["labels"]
    cuts = [lab["best_gap"], lab["inaccuracy_drop"], lab["mistake_drop"], lab["blunder_drop"]]
    end = cuts[-1] + 6
    left, right = 40, W - 40
    y0, h = 58, 54

    def x(v: float) -> float:
        return left + v / end * (right - left)

    names = ["best", "good", "inaccuracy", "mistake", "blunder"]
    bounds = [0] + cuts + [end]
    body = [heading(left, 30, "MOVE LABELS · drop in win probability, from the mover's view (win points)", t=t)]
    for i, name in enumerate(names):
        fill, stroke, ink = t["labels"][i]
        x1, x2 = x(bounds[i]), x(bounds[i + 1])
        body.append(f'<rect x="{x1:g}" y="{y0}" width="{x2 - x1:g}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        body.append(text((x1 + x2) / 2, y0 + 33, name, fill=ink, size=13, weight="700"))
    for v in [0] + cuts:
        body.append(f'<line x1="{x(v):g}" y1="{y0 + h}" x2="{x(v):g}" y2="{y0 + h + 6}" stroke="{t["muted"]}"/>')
        body.append(text(x(v), y0 + h + 20, str(v), fill=t["muted"], size=11.5, mono=True))
    body.append(text(x(end) - 6, y0 + h + 20, "…", fill=t["muted"], size=12.5, anchor="end"))
    body.append(text(left, y0 - 8, "top move", fill=t["muted"], size=12, anchor="start"))
    body.append(text(right, y0 - 8, "larger loss ▶", fill=t["muted"], size=12, anchor="end"))
    body.append(text(left, y0 + h + 48, f"best: the top move, or within {cuts[0]} win points of it. A drop exactly on a cutoff takes the higher label.", fill=t["muted"], size=12, anchor="start"))
    desc = (
        "Move labels by the drop in win probability from the mover's view, in win points: best is the top move or within "
        f"{cuts[0]} of it, good up to {cuts[1]}, inaccuracy from {cuts[1]}, mistake from {cuts[2]}, blunder from {cuts[3]}. "
        "A drop exactly on a cutoff takes the higher label."
    )
    return svg(height=146, title="Move labels", desc=desc, body=body, t=t)


def scores(*, cfg: dict, t: dict) -> str:
    s = cfg["coverage_saturation"]
    k = cfg["precision_min_checked"]
    top, bottom = 64, 214
    body = [heading(40, 30, "HOW COVERAGE AND PRECISION GROW", t=t)]

    def axes(x0: float, x1: float, ylab: str) -> list[str]:
        out = [f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="{t["muted"]}"/>', f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="{t["muted"]}"/>']
        for frac, lab in ((0, "0"), (0.5, "0.5"), (1, "1")):
            yy = bottom - frac * (bottom - top)
            if frac:
                out.append(f'<line x1="{x0}" y1="{yy:g}" x2="{x1}" y2="{yy:g}" stroke="{t["grid"]}" stroke-dasharray="3 3"/>')
            out.append(text(x0 - 8, yy + 4, lab, fill=t["muted"], size=11.5, anchor="end", mono=True))
        out.append(text(x0, top - 14, ylab, fill=t["text"], size=13, anchor="start", weight="700"))
        return out

    # left panel: C against the share of the expert points' weight earned
    x0, x1 = 80, 420

    def cx(v: float) -> float:
        return x0 + v * (x1 - x0)

    def yy(v: float) -> float:
        return bottom - v * (bottom - top)

    body += axes(x0, x1, "Coverage C")
    body.append(f'<polyline points="{cx(0):g},{yy(0):g} {cx(s):g},{yy(1):g} {cx(1):g},{yy(1):g}" fill="none" stroke="{t["line"]}" stroke-width="2.5"/>')
    for v in (0, 0.2, 0.4, s, 0.8, 1):
        body.append(f'<line x1="{cx(v):g}" y1="{bottom}" x2="{cx(v):g}" y2="{bottom + 5}" stroke="{t["muted"]}"/>')
        body.append(text(cx(v), bottom + 18, f"{v:.0%}", fill=t["muted"], size=11.5, mono=True))
    body.append(f'<line x1="{cx(s):g}" y1="{yy(1):g}" x2="{cx(s):g}" y2="{bottom}" stroke="{t["grid"]}" stroke-dasharray="3 3"/>')
    body.append(text(cx(s) + 8, yy(1) + 18, f"full from {s:.0%} of the weight", fill=t["text"], size=12.5, anchor="start"))
    body.append(text((x0 + x1) / 2, bottom + 38, "share of the expert points' weight the answer earns", fill=t["muted"], size=12))

    # right panel: the ceiling on P against the number of checked claims
    x0, x1 = 540, 860
    n = 6
    body += axes(x0, x1, "Most P can be")
    step = (x1 - x0) / (n + 1)
    fill, stroke, _ = t["bands"][0]
    for c in range(n + 1):
        ceiling = min(1, c / k)
        bx = x0 + step * (c + 0.5) - 14
        if ceiling:
            body.append(f'<rect x="{bx:g}" y="{yy(ceiling):g}" width="28" height="{bottom - yy(ceiling):g}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        body.append(text(bx + 14, bottom + 18, str(c), fill=t["muted"], size=11.5, mono=True))
    body.append(text((x0 + x1) / 2, bottom + 38, "claims checked (true, false or illegal)", fill=t["muted"], size=12))
    body.append(text((x0 + x1) / 2, bottom + 58, f"below {k} checked, P is at most checked ÷ {k}", fill=t["text"], size=12.5))
    desc = (
        f"Left: coverage C rises in proportion to the share of the expert points' weight the answer earns and is full (1) from {s:.0%} of it. "
        f"Right: the most precision P can be for a given number of checked claims: 0 with none, checked divided by {k} below {k}, "
        f"and 1 from {k} on, where P is the plain share of true claims."
    )
    return svg(height=292, title="Coverage and precision", desc=desc, body=body, t=t)


def main(*, out: Path = OUT) -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for name, draw in (("eval-bands", eval_bands), ("move-labels", move_labels), ("scores", scores)):
        for theme, t in THEMES.items():
            (out / f"{name}-{theme}.svg").write_text(draw(cfg=cfg, t=t), encoding="utf-8")


if __name__ == "__main__":
    main()
