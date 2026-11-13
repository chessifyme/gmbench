from bench.config import BenchConfig, LabelThresholds, load_bench, private_block

FIXTURE_BARS = LabelThresholds(blunder_drop=30.0, mistake_drop=20.0, inaccuracy_drop=10.0, best_gap=2.0)


def fixture_cfg() -> BenchConfig:
    """The repository config with the 10 / 20 / 30 bars that the drops and stored labels under tests/fixtures were built around."""
    cfg = load_bench()
    bars = cfg.turning_points.model_copy(update={"candidate_min_drop": 10.0, "sheet_min_drop": 20.0, "missed_win_min_drop": 10.0})
    # the maintainers' selection block exists only where their private overlay sets it
    selection = {"selection": private_block(value=cfg.selection, name="selection").model_copy(update={"prefilter_margin": 5.0})} if cfg.selection is not None else {}
    return cfg.model_copy(update={"labels": FIXTURE_BARS, "turning_points": bars, **selection})
