import pytest

from bench.bands import BAND_ORDER, band_cp_range, band_holds, bands_compatible
from bench.config import EvalBands, load_bench
from bench.drafting import BAND_TEXT, band_for_cp
from bench.judge_tools import eval_band
from bench.models import Band

BANDS = load_bench().eval_bands

# the benchmark's scale in pawns from either side: 0-0.3 roughly equal, 0.3-0.7 slight edge, 0.7-1.5 clear edge, 1.5-2.0 large edge, 2.0+ winning
EDGES = [
    (0, "equal"),
    (29, "equal"),
    (30, "white_slightly_better"),
    (31, "white_slightly_better"),
    (69, "white_slightly_better"),
    (70, "white_better"),
    (71, "white_better"),
    (149, "white_better"),
    (150, "white_large_edge"),
    (151, "white_large_edge"),
    (199, "white_large_edge"),
    (200, "white_winning"),
    (201, "white_winning"),
    (1000, "white_winning"),
]


def test_the_config_holds_the_leads_scale() -> None:
    assert (BANDS.slightly_better_cp, BANDS.better_cp, BANDS.large_edge_cp, BANDS.winning_cp, BANDS.margin_cp) == (30, 70, 150, 200, 15)
    assert load_bench().stated_eval_cap_cp == BANDS.winning_cp, "a stated number is capped where the winning band starts"


@pytest.mark.parametrize(("cp", "band"), EDGES + [(-cp, band.replace("white_", "black_")) for cp, band in EDGES if cp])
def test_an_evaluation_falls_in_the_same_band_for_the_drafter_and_the_judge_tools(cp: int, band: str) -> None:
    assert band_for_cp(cp_white=cp, bands=BANDS) == band
    assert eval_band(cp_white=cp, bands=BANDS) == band
    low, high = band_cp_range(band=band, bands=BANDS)
    assert low <= cp <= high


def test_there_are_nine_bands_in_order_and_each_has_drafting_words() -> None:
    from typing import get_args

    from bench.grading.claims import EVAL_BANDS

    assert BAND_ORDER == (
        "black_winning",
        "black_large_edge",
        "black_better",
        "black_slightly_better",
        "equal",
        "white_slightly_better",
        "white_better",
        "white_large_edge",
        "white_winning",
    )
    assert set(get_args(Band)) == set(BAND_ORDER) == set(BAND_TEXT) == set(EVAL_BANDS)
    assert BAND_TEXT["white_large_edge"] == "White has a large advantage" and BAND_TEXT["black_better"] == "Black is clearly better"
    ranges = [band_cp_range(band=b, bands=BANDS) for b in BAND_ORDER]
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])), "the bands tile the scale"


def test_band_holds_inside_the_band_widened_by_the_margin() -> None:
    assert BANDS.margin_cp == 15
    assert band_holds(band="equal", cp_white=-45, bands=BANDS) and band_holds(band="equal", cp_white=45, bands=BANDS)
    assert not band_holds(band="equal", cp_white=46, bands=BANDS) and not band_holds(band="equal", cp_white=-46, bands=BANDS)
    assert band_holds(band="white_slightly_better", cp_white=15, bands=BANDS) and not band_holds(band="white_slightly_better", cp_white=14, bands=BANDS)
    assert band_holds(band="white_slightly_better", cp_white=85, bands=BANDS) and not band_holds(band="white_slightly_better", cp_white=86, bands=BANDS)
    assert band_holds(band="white_better", cp_white=55, bands=BANDS) and not band_holds(band="white_better", cp_white=54, bands=BANDS)
    assert band_holds(band="white_better", cp_white=165, bands=BANDS) and not band_holds(band="white_better", cp_white=166, bands=BANDS)
    assert band_holds(band="white_large_edge", cp_white=135, bands=BANDS) and not band_holds(band="white_large_edge", cp_white=134, bands=BANDS)
    assert band_holds(band="white_large_edge", cp_white=215, bands=BANDS) and not band_holds(band="white_large_edge", cp_white=216, bands=BANDS)
    assert band_holds(band="white_winning", cp_white=185, bands=BANDS) and band_holds(band="white_winning", cp_white=1000, bands=BANDS)
    assert not band_holds(band="white_winning", cp_white=184, bands=BANDS)


def test_the_margin_never_reaches_two_bands_away() -> None:
    """With 15 cp of margin a band claim fits at most its two neighbours. The slight-edge band is 40 cp wide: the old 30 cp margin on
    these edges would have let "slightly better" hold at 0.00 and "equal" at +0.60."""
    for index, band in enumerate(BAND_ORDER):
        far = BAND_ORDER[: max(0, index - 1)] + BAND_ORDER[index + 2 :]
        for other in far:
            assert not bands_compatible(claimed=band, sheet_band=other, bands=BANDS), (band, other)
            low, high = band_cp_range(band=other, bands=BANDS)
            middle = 1000 if high == float("inf") else -1000 if low == float("-inf") else int((low + high) / 2)
            assert not band_holds(band=band, cp_white=middle, bands=BANDS), (band, other, middle)


def test_band_holds_is_symmetric_for_black() -> None:
    for band in BAND_ORDER:
        mirrored = band.replace("black_", "white_") if band.startswith("black_") else band.replace("white_", "black_")
        for cp in (-1000, -216, -215, -200, -185, -184, -150, -135, -134, -70, -55, -54, -46, -45, -30, -15, -14, 0, 14, 15, 30, 45, 46, 85, 86, 165, 166, 215, 216, 1000):
            assert band_holds(band=band, cp_white=cp, bands=BANDS) == band_holds(band=mirrored, cp_white=-cp, bands=BANDS)


def test_band_far_from_the_reference_is_false_even_when_it_is_the_neighbouring_band() -> None:
    assert not band_holds(band="equal", cp_white=60, bands=BANDS)
    assert band_holds(band="white_slightly_better", cp_white=60, bands=BANDS)
    assert not band_holds(band="white_better", cp_white=180, bands=BANDS) and band_holds(band="white_large_edge", cp_white=180, bands=BANDS)


def test_unknown_band_names_raise() -> None:
    with pytest.raises(ValueError, match="not an evaluation band"):
        band_holds(band="white_crushing", cp_white=0, bands=BANDS)


def test_bands_compatible_follows_the_same_margin() -> None:
    assert bands_compatible(claimed="equal", sheet_band="white_slightly_better", bands=BANDS)
    assert bands_compatible(claimed="white_better", sheet_band="white_large_edge", bands=BANDS)
    assert bands_compatible(claimed="white_large_edge", sheet_band="white_winning", bands=BANDS)
    assert not bands_compatible(claimed="white_better", sheet_band="white_winning", bands=BANDS)
    assert not bands_compatible(claimed="equal", sheet_band="white_better", bands=BANDS)
    assert not bands_compatible(claimed="white_slightly_better", sheet_band="black_slightly_better", bands=BANDS)
    assert bands_compatible(claimed="white_winning", sheet_band="white_winning", bands=BANDS)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"slightly_better_cp": 30, "better_cp": 30, "large_edge_cp": 150, "winning_cp": 200, "margin_cp": 15}, "must rise strictly"),
        ({"slightly_better_cp": 30, "better_cp": 70, "large_edge_cp": 250, "winning_cp": 200, "margin_cp": 15}, "must rise strictly"),
        ({"slightly_better_cp": 30, "better_cp": 70, "large_edge_cp": 150, "winning_cp": 200, "margin_cp": 20}, "under half the narrowest band"),
        ({"slightly_better_cp": 30, "better_cp": 70, "large_edge_cp": 150, "winning_cp": 200, "margin_cp": -1}, "under half the narrowest band"),
    ],
)
def test_the_config_refuses_edges_out_of_order_and_a_margin_of_half_a_band_or_more(fields: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvalBands.model_validate(fields)
