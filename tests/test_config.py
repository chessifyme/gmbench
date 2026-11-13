from pathlib import Path
import json
import pytest
from pydantic import ValidationError

from bench.config import BenchConfig, CalibrationRules, EngineEnv, EntrantModel, ModelSpec, ModelsConfig, SelectionRules, SelectRules, Weights, private_block, load_bench, load_models, load_models_file, models_profile, vendor_for


def test_bench_config_has_spec_values() -> None:
    cfg = load_bench()
    assert cfg.reference_nodes_position == 5_000_000
    assert cfg.reference_nodes_ply == 1_000_000
    assert cfg.answer_timeout_s == 900
    assert cfg.runs_per_task == 3
    assert (cfg.weights.coverage, cfg.weights.precision, cfg.weights.coverage_without_family, cfg.weights.precision_without_family) == (0.5, 0.5, 0.5, 0.5)
    assert (cfg.weights.verdict, cfg.weights.audience, cfg.weights.illegal_move_per_move, cfg.weights.illegal_move_cap) == (0.0, 0.0, 0.0, 0.0)
    assert (cfg.weights.game_review_coverage, cfg.weights.moments, cfg.weights.moment_labels) == (1.0, 0.0, 0.0)
    assert "penalty_per_point" not in cfg.weights.model_dump() and "penalty_cap" not in cfg.weights.model_dump()
    assert (cfg.labels.inaccuracy_drop, cfg.labels.mistake_drop, cfg.labels.blunder_drop, cfg.labels.best_gap) == (5, 10, 15, 2)  # Lichess, lila Advice.scala
    assert cfg.eval_bands.slightly_better_cp == 30
    assert cfg.eval_bands.better_cp == 70
    assert cfg.eval_bands.large_edge_cp == 150
    assert cfg.eval_bands.winning_cp == 200
    assert cfg.eval_bands.margin_cp == 15
    assert cfg.engine_tolerance_win_pct == 5.0
    assert cfg.stated_eval_cap_cp == 200
    assert cfg.dataset_version == "v0"
    assert cfg.family_counts == {"assessment_plan": 15, "move_choice": 15, "move_judgement": 15, "game_review": 0}
    assert cfg.run_status_allowed == ["draft", "reviewed", "signed"]
    assert cfg.publish_status_allowed == ["signed"]
    assert cfg.engine.version_expected == "Stockfish 19"


def test_jury_is_per_item_with_one_fixed_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCH_PROFILE", raising=False)
    cfg = load_bench()
    assert cfg.jury_mode == "per_item" and cfg.jury_size == 1 and cfg.claim_jury_size == 1


def test_v0_headline_is_the_three_position_families_without_game_review() -> None:
    assert load_bench().headline_families == ["assessment_plan", "move_choice", "move_judgement"]


def test_game_review_weights_sum_to_one() -> None:
    w = load_bench().weights
    assert w.game_review_coverage + w.moments + w.moment_labels == pytest.approx(1.0)
    with pytest.raises(ValueError, match="game_review_coverage"):
        Weights.model_validate({**w.model_dump(), "moments": 0.5})


def test_turning_point_and_select_rules_for_the_new_grader() -> None:
    cfg = load_bench()
    assert cfg.turning_points.sheet_min_drop == 10 and cfg.turning_points.max_named_moments == 8
    assert cfg.turning_points.candidate_min_drop == 5 and cfg.turning_points.missed_win_min_drop == 5
    assert cfg.select.move_judgement_label_shares == {"best": 0.25, "inaccuracy": 0.25, "mistake": 0.25, "blunder": 0.25}
    assert not hasattr(cfg.select, "move_judgement_bad_share")


def test_label_shares_must_cover_every_label_and_sum_to_one() -> None:
    rules = load_bench().select.model_dump()
    with pytest.raises(ValidationError, match="sum to 1"):
        SelectRules.model_validate({**rules, "move_judgement_label_shares": {"best": 0.5, "inaccuracy": 0.25, "mistake": 0.25, "blunder": 0.25}})
    with pytest.raises(ValidationError, match="exactly"):
        SelectRules.model_validate({**rules, "move_judgement_label_shares": {"best": 0.5, "blunder": 0.5}})




def test_the_blocks_only_the_maintainers_tools_read_are_not_in_config_bench_yaml() -> None:
    from bench.config import CONFIG_DIR

    import yaml

    bench = yaml.safe_load((CONFIG_DIR / "bench.yaml").read_text(encoding="utf-8"))
    assert not {"pool", "selection", "calibration_sampling"} & set(bench) and set(bench["select"]) == {"move_judgement_label_shares"}
    with pytest.raises(RuntimeError, match="no pool: block"):
        private_block(value=None, name="pool")


def test_engine_context_calibration_and_governance_blocks() -> None:
    cfg = load_bench()
    assert (cfg.engine_context.nodes, cfg.engine_context.multipv, cfg.engine_context.nodes_per_ply) == (1_000_000, 3, 200_000)
    assert cfg.governance.maintainer == "Chessify"
    assert cfg.governance.grandmaster is None and cfg.governance.grandmaster_affiliation is None
    assert "Chessify maintains this benchmark and also builds Ziuk" in cfg.governance.conflict_of_interest
    assert "Ziuk may use analysis tools or services" in cfg.governance.conflict_of_interest and "Ziuk uses" not in cfg.governance.conflict_of_interest
    assert "pre-annotated from Stockfish and a mix of AI models and agents" in cfg.governance.conflict_of_interest and "%" not in cfg.governance.conflict_of_interest


def test_the_pair_tie_margin_is_a_non_negative_score_gap() -> None:
    rules = load_bench().calibration
    assert rules.pair_tie_margin == 5.0
    with pytest.raises(ValidationError, match="pair_tie_margin"):
        CalibrationRules.model_validate({**rules.model_dump(), "pair_tie_margin": -1.0})


def test_ratings_are_anchored_at_a_field_mean_of_1500() -> None:
    assert load_bench().calibration.rating_anchor == 1500.0


def test_removed_weight_fields_are_rejected() -> None:
    weights = load_bench().weights.model_dump()
    for old in ("family", "penalty", "penalty_per_point", "penalty_cap", "hallucination"):
        with pytest.raises(ValidationError):
            Weights.model_validate({**weights, old: 0.1})


def test_engine_env_defaults_to_stockfish_on_path() -> None:
    assert EngineEnv(_env_file=None).engine_path == "stockfish"


def test_every_published_grading_role_is_a_fixed_model_of_one_of_three_families_over_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCH_PROFILE", raising=False)
    models = load_models()
    assert {"gemini_default", "openai_default", "claude_default"} <= set(models.entrants)
    assert models.entrants["gemini_default"].family == "google"
    panel = {"openai": ("openai/gpt-6.1-terra", ["OpenAI"]), "anthropic": ("anthropic/claude-sonnet-5.5", ["Anthropic"]), "google": ("google/gemini-3.8-flash", ["Google AI Studio"])}
    for specs in (models.judges, models.claim_judges):
        assert [(key, spec.model_id, spec.provider) for key, spec in specs.items()] == [(key, *value) for key, value in panel.items()]
    assert {key: (spec.model_id, spec.provider) for key, spec in models.extractors.items()} == {"openai": panel["openai"]}
    for spec in (*models.judges.values(), *models.extractors.values(), *models.claim_judges.values()):
        assert (spec.transport, spec.effort) == ("openrouter_api", "high")
    assert len(models.judges) >= load_bench().jury_size and len(models.claim_judges) >= load_bench().claim_jury_size
    assert not hasattr(models, "extractor")


def test_entrants_include_engine_baselines_and_calibration_entrants() -> None:
    models = load_models()
    assert {name for name, e in models.entrants.items() if e.engine_context} == {"gemini_engine", "openai_engine", "claude_engine", "claude_openrouter_engine", "gemini_openrouter_engine"}
    assert all(not e.engine_context for name, e in models.entrants.items() if name.endswith("_default"))
    for name in ("gemini", "openai", "claude"):
        assert models.entrants[f"{name}_engine"].model_id == models.entrants[f"{name}_default"].model_id
    assert set(models.calibration_entrants) == {"calibration_haiku", "calibration_grok"}
    assert models.calibration_entrants["calibration_grok"].family == "xai"
    assert not set(models.calibration_entrants) & set(models.entrants)


def test_vendor_for_grok_is_xai() -> None:
    assert load_models().vendor_for(model_id="grok-4") == "xai"


def test_models_config_vendor_for_unknown_prefix_raises() -> None:
    with pytest.raises(ValueError):
        load_models().vendor_for(model_id="llama-9")


def test_vendor_for_composer_is_cursor() -> None:
    assert load_models().vendor_for(model_id="composer-2.5") == "cursor"


def test_vendor_for_cursor_prefixed_grok_is_xai() -> None:
    assert load_models().vendor_for(model_id="cursor-grok-4.6-low-fast") == "xai"


def test_claude_only_profile_runs_every_role_through_claude_code(monkeypatch: pytest.MonkeyPatch) -> None:
    models = load_models(profile="claude_only")
    assert list(models.judges) == ["anthropic_opus"]
    assert {spec.transport for spec in models.judges.values()} == {"claude_cli"}
    assert {models.vendor_for(model_id=spec.model_id) for spec in models.judges.values()} == {"anthropic"}
    assert list(models.extractors) == ["anthropic_sonnet", "anthropic_haiku"]
    assert {spec.transport for spec in models.extractors.values()} == {"claude_cli"}
    assert set(models.entrants) == {"claude_default", "claude_engine", "claude_tools", "claude_opus_tools", "claude_openrouter", "claude_openrouter_engine", "claude_coach_tools"}
    assert models.entrants["claude_default"].engine_context is False and models.entrants["claude_engine"].engine_context is True
    assert set(models.calibration_entrants) == {"calibration_haiku"}
    monkeypatch.setenv("BENCH_PROFILE", "claude_only")
    assert models_profile() == "claude_only" and load_models().judges == models.judges
    monkeypatch.delenv("BENCH_PROFILE")
    assert models_profile() is None and "openai" in load_models().judges


def test_unknown_profile_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_PROFILE", "gpt-only")
    with pytest.raises(ValueError, match="BENCH_PROFILE"):
        models_profile()


def test_judge_keys_must_be_underscore_names() -> None:
    with pytest.raises(ValueError, match="underscore"):
        ModelsConfig.model_validate({**load_models().model_dump(), "judges": {"anthropic-opus": {"model_id": "claude-opus-5-5", "transport": "claude_cli", "effort": "high"}}})


def test_extractor_keys_must_be_underscore_names() -> None:
    with pytest.raises(ValueError, match="underscore"):
        ModelsConfig.model_validate({**load_models().model_dump(), "extractors": {"anthropic-sonnet": {"model_id": "claude-sonnet-5-5", "transport": "claude_cli", "effort": "high"}}})


def test_positive_weights_sum_to_one_with_and_without_the_verdict_term() -> None:
    w = load_bench().weights
    assert w.coverage + w.precision + w.verdict + w.audience == pytest.approx(1.0)
    assert w.coverage_without_family + w.precision_without_family + w.audience == pytest.approx(1.0)
    with pytest.raises(ValueError, match="sum to 1"):
        Weights.model_validate({**w.model_dump(), "audience": 0.2})
    with pytest.raises(ValueError, match="sum to 1"):
        Weights.model_validate({**w.model_dump(), "precision_without_family": 0.6})


def test_claude_only_jury_is_opus_alone_and_haiku_stays_an_extractor_and_calibration_entrant() -> None:
    models = load_models(profile="claude_only")
    assert list(models.judges) == ["anthropic_opus"] and models.judges["anthropic_opus"].model_id == "claude-opus-5-5"
    assert len(models.judges) == models.jury_size == models.claim_jury_size == 1
    assert "anthropic_haiku" in models.extractors and models.calibration_entrants["calibration_haiku"].model_id == "claude-haiku-5-5"


def test_every_claude_code_role_names_its_effort_in_both_profiles() -> None:
    for profile in (None, "claude_only"):
        models = load_models(profile=profile)
        graders = [*models.judges.values(), *models.extractors.values(), *models.claim_judges.values(), *models.calibration_entrants.values()]
        entrants = list(models.entrants.values())
        specs = [*graders, *entrants]
        assert {spec.effort for spec in graders if spec.transport in ("claude_cli", "openrouter_api")} == {"high"}
        assert {spec.effort for spec in entrants if spec.transport in ("claude_cli", "openrouter_api")} == {"medium"}
        assert all(spec.effort is None for spec in specs if spec.transport not in ("claude_cli", "openrouter_api"))


def test_with_tools_entrants_are_declared_with_labels() -> None:
    models = load_models()
    assert models.entrants["claude_tools"].tools is True and models.entrants["claude_tools"].model_id == "claude-sonnet-5-5"
    assert models.entrants["openai_tools"].tools is True and models.entrants["openai_tools"].model_id == "gpt-6-luna"
    assert models.entrants["claude_default"].tools is False
    assert models.entrants["claude_default"].label == "Claude Opus 5.5" and models.entrants["openai_default"].label == "GPT-6.1 Sol"
    assert models.entrants["claude_tools"].label.endswith("with tools") and models.entrants["openai_tools"].label.endswith("with tools")
    profile = load_models(profile="claude_only")
    assert profile.entrants["claude_tools"].tools is True and profile.entrants["claude_default"].tools is False
    for config in (models, profile):
        opus = config.entrants["claude_opus_tools"]
        assert opus.tools is True and opus.model_id == "claude-opus-5-5" and opus.transport == "claude_cli" and opus.engine_context is False
        assert opus.label == "Claude Opus 5.5, with tools"


def test_calibrated_weights_must_match_bench_yaml_once_they_exist(tmp_path: Path) -> None:
    from bench.config import assert_weights_frozen

    cfg = load_bench()
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    assert load_bench(calibration_dir=calibration).weights == cfg.weights
    path = calibration / f"weights_{cfg.dataset_version}.json"
    path.write_text(json.dumps({"weights": cfg.weights.model_dump()}), encoding="utf-8")
    assert_weights_frozen(cfg=cfg, calibration_dir=calibration)
    path.write_text(json.dumps({"weights": {**cfg.weights.model_dump(), "coverage": 0.60, "precision": 0.40, "coverage_without_family": 0.60, "precision_without_family": 0.40}}), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage, coverage_without_family, precision, precision_without_family"):
        load_bench(calibration_dir=calibration)


def test_a_reference_is_labelled_with_the_label_thresholds_and_the_candidate_bar() -> None:
    cfg = load_bench()
    labelling = cfg.reference_labelling()
    assert labelling.labels == cfg.labels
    assert labelling.candidate_min_drop == cfg.turning_points.candidate_min_drop
    assert labelling.model_dump() == {"labels": {"blunder_drop": 15, "mistake_drop": 10, "inaccuracy_drop": 5, "best_gap": 2}, "candidate_min_drop": 5, "winning_cp": 200}
    stricter = cfg.model_copy(update={"labels": cfg.labels.model_copy(update={"inaccuracy_drop": 4.0})})
    assert stricter.reference_labelling() != labelling


@pytest.mark.parametrize(
    ("model_id", "vendor"),
    [
        ("anthropic/claude-opus-5.5", "anthropic"),
        ("google/gemini-4-argon", "google"),
        ("openai/gpt-6.1-sol", "openai"),
        ("openai/o5-mini", "openai"),
        ("x-ai/grok-4.7", "xai"),
        ("o3", "openai"),
        ("o4-mini", "openai"),
        ("gpt-6.1-sol", "openai"),
        ("claude-opus-5-5", "anthropic"),
        ("cursor-grok-4.5-low-fast", "xai"),
    ],
)
def test_vendor_for_reads_openrouter_ids_and_bare_ids(model_id: str, vendor: str) -> None:
    assert vendor_for(model_id=model_id) == vendor


@pytest.mark.parametrize("model_id", ["openrouter/auto", "omni-1", "o", "ollama/gpt-oss", "meta-llama/llama-5", "anthropic/llama-9", "google/claude-opus-5.5"])
def test_vendor_for_refuses_ids_it_cannot_place(model_id: str) -> None:
    with pytest.raises(ValueError, match="vendor"):
        vendor_for(model_id=model_id)


def test_an_openrouter_spec_pins_its_provider_and_names_its_effort() -> None:
    spec = ModelSpec(model_id="anthropic/claude-opus-5.5", transport="openrouter_api", provider=["Anthropic"], effort="medium", max_tokens=16000)
    assert spec.provider == ["Anthropic"] and spec.temperature is None and spec.seed is None and spec.max_tokens == 16000
    with pytest.raises(ValidationError, match="provider"):
        ModelSpec(model_id="anthropic/claude-opus-5.5", transport="openrouter_api", effort="medium")
    with pytest.raises(ValidationError, match="provider"):
        ModelSpec(model_id="anthropic/claude-opus-5.5", transport="openrouter_api", provider=[], effort="medium")
    with pytest.raises(ValidationError, match="effort"):
        ModelSpec(model_id="anthropic/claude-opus-5.5", transport="openrouter_api", provider=["Anthropic"])


@pytest.mark.parametrize("option", [{"provider": ["Anthropic"]}, {"temperature": 1.0}, {"seed": 7}, {"max_tokens": 100}])
def test_api_options_are_refused_on_a_cli_transport(option: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="openrouter_api"):
        ModelSpec.model_validate({"model_id": "claude-opus-5-5", "transport": "claude_cli", "effort": "high", **option})


def test_an_entrant_hands_its_model_spec_with_every_api_option() -> None:
    entrant = EntrantModel(family="google", model_id="google/gemini-4-argon", transport="openrouter_api", provider=["Google AI Studio"], effort="medium", temperature=1.0, seed=7, max_tokens=16000, label="Gemini 4 Argon (API)", engine_context=False)
    assert entrant.model_spec() == ModelSpec(model_id="google/gemini-4-argon", transport="openrouter_api", provider=["Google AI Studio"], effort="medium", temperature=1.0, seed=7, max_tokens=16000)


def test_openrouter_entrants_in_both_profiles() -> None:
    for profile in (None, "claude_only"):
        claude = load_models(profile=profile).entrants["claude_openrouter"]
        assert (claude.family, claude.model_id, claude.transport, claude.provider, claude.effort, claude.max_tokens) == ("anthropic", "anthropic/claude-opus-5.5", "openrouter_api", ["Anthropic"], "medium", 16000)
        assert claude.label == "Claude Opus 5.5 (API)" and claude.engine_context is False and claude.temperature is None and claude.seed is None
    gemini = load_models().entrants["gemini_openrouter"]
    assert (gemini.family, gemini.model_id, gemini.transport, gemini.provider, gemini.effort, gemini.max_tokens, gemini.temperature) == ("google", "google/gemini-4-argon", "openrouter_api", ["Google AI Studio"], "medium", 16000, None)
    assert gemini.label == "Gemini 4 Argon (API)" and "gemini_openrouter" not in load_models(profile="claude_only").entrants


def test_review_adjudicators_are_a_comma_separated_list_of_names() -> None:
    from bench.config import ReviewEnv

    assert ReviewEnv(review_adjudicators=" Aram Hakobyan, Bella ,,").adjudicator_names() == frozenset({"Aram Hakobyan", "Bella"})
    assert ReviewEnv(review_adjudicators="").adjudicator_names() == frozenset()



def test_a_claim_jury_needs_at_least_one_claim_judge() -> None:
    raw = load_bench().model_dump()
    with pytest.raises(ValidationError, match="claim_jury_size"):
        BenchConfig.model_validate({**raw, "claim_jury_size": 0})


def test_the_v1_api_entrants_have_engine_analysis_twins_on_the_same_pinned_route() -> None:
    entrants = load_models_file(profile=None).entrants
    for plain, engine in (("claude_openrouter", "claude_openrouter_engine"), ("gemini_openrouter", "gemini_openrouter_engine")):
        assert entrants[engine].engine_context and not entrants[plain].engine_context
        assert entrants[engine].model_copy(update={"engine_context": False, "label": entrants[plain].label}) == entrants[plain]


def test_a_profile_may_set_its_own_panel_sizes_and_the_published_judge_is_gpt_terra_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCH_PROFILE", raising=False)
    published = load_bench()
    assert (published.jury_size, published.claim_jury_size) == (1, 1)
    models = load_models_file(profile=None)
    assert [vendor_for(model_id=m.model_id) for m in models.judges.values()] == ["openai", "anthropic", "google"]
    assert [m.model_id for m in models.claim_judges.values()] == [m.model_id for m in models.judges.values()]
    jury = [m.model_id for m in list(models.judges.values())[: published.jury_size]]
    claim_jury = [m.model_id for m in list(models.claim_judges.values())[: published.claim_jury_size]]
    assert jury == claim_jury == ["openai/gpt-6.1-terra"]
    assert len(models.judges) == len(models.claim_judges) == 3
    assert [m.model_id for m in models.extractors.values()] == ["openai/gpt-6.1-terra"]
    assert all(m.transport == "openrouter_api" and m.provider for m in [*models.judges.values(), *models.extractors.values(), *models.claim_judges.values()])
    monkeypatch.setenv("BENCH_PROFILE", "claude_only")
    development = load_bench()
    assert (development.jury_size, development.claim_jury_size) == (1, 1)
