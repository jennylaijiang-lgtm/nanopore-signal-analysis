"""Selected original tests for the frozen-split DBA preprocessing ablation.

Extracted from the shared project test module; HMM and plotting tests omitted.
"""
from dataclasses import replace
import numpy as np
import scripts.evaluate_psk_preprocessing_ablation as runner
from scripts.evaluate_psk_dba_consensus import PreparedEvent, SignalTrace

def _event(
    dataset: str,
    event_id: int,
    *,
    stored_is_consensus: bool,
    length: int,
    is_training: bool | None = None,
) -> PreparedEvent:
    trace = SignalTrace(
        mean=np.linspace(0.2, 0.8, length, dtype=np.float64),
        std=np.full(length, 0.05, dtype=np.float64),
        dwell=np.ones(length, dtype=np.float64),
    )
    return PreparedEvent(
        dataset=dataset,
        event_id=event_id,
        event_order=event_id,
        stored_is_consensus=stored_is_consensus,
        is_training=(
            not stored_is_consensus if is_training is None else is_training
        ),
        peptide=trace,
        dna_window=trace,
        tail=None,
        dna_shift=0.0,
        dna_scale=1.0,
        nuisance=np.zeros(5),
        tail_steps=0,
        raw_peptide_steps=length,
    )


def test_freeze_stored_split_never_uses_active_assignment() -> None:
    events = {
        "JS445": [
            _event("JS445", 1, stored_is_consensus=True, length=5),
            _event("JS445", 2, stored_is_consensus=False, length=5),
        ]
    }

    frozen = runner.freeze_stored_split(events)

    assert frozen["JS445"][0].is_training
    assert not frozen["JS445"][1].is_training
    assert [event.stored_is_consensus for event in frozen["JS445"]] == [True, False]


def test_p95_target_uses_only_stored_training_lengths() -> None:
    events = {
        "JS445": [
            _event("JS445", 1, stored_is_consensus=True, length=10),
            _event("JS445", 2, stored_is_consensus=True, length=12),
            _event("JS445", 3, stored_is_consensus=False, length=1000),
        ],
        "JS446": [
            _event("JS446", 1, stored_is_consensus=True, length=8),
            _event("JS446", 2, stored_is_consensus=True, length=9),
            _event("JS446", 3, stored_is_consensus=False, length=2000),
        ],
    }

    target, values, determining = runner.select_p95_target(events, percentile=95.0)

    assert target == 12
    assert values["JS445"] == np.percentile([10, 12], 95.0, method="linear")
    assert values["JS446"] == np.percentile([8, 9], 95.0, method="linear")
    assert determining == ("JS445",)


def test_length_normalization_preserves_identity_and_master_split() -> None:
    event = _event("JS445", 7, stored_is_consensus=False, length=5)

    transformed = runner.length_normalize_events(
        {"JS445": [event]}, target_length=11, min_std=1e-3
    )["JS445"][0]

    assert transformed.event_id == event.event_id
    assert transformed.stored_is_consensus is event.stored_is_consensus
    assert transformed.is_training is event.is_training
    assert transformed.peptide.mean.size == 11
    assert transformed.peptide.std.size == 11
    np.testing.assert_array_equal(transformed.peptide.dwell, np.ones(11))


def test_common_heldout_keys_intersects_filters_without_training_events() -> None:
    source = {
        "JS445": [
            _event("JS445", 1, stored_is_consensus=True, length=5),
            _event("JS445", 2, stored_is_consensus=False, length=5),
            _event("JS445", 3, stored_is_consensus=False, length=5),
        ]
    }
    native = runner.Condition("TO", "TO", "none", "native", None, source)
    normalized_events = {
        "JS445": [source["JS445"][0], source["JS445"][2]]
    }
    normalized = runner.Condition(
        "TNO", "TNO", "fixed_100", "fixed_100", 100, normalized_events
    )

    assert runner.common_heldout_keys([native, normalized]) == {("JS445", 3)}


def test_screen_restores_stored_split_after_filter(monkeypatch: object) -> None:
    source_event = _event("JS445", 1, stored_is_consensus=True, length=5)
    overwritten = replace(source_event, is_training=False)

    def fake_filter(*args: object, **kwargs: object) -> tuple[dict[str, list[PreparedEvent]], list[dict[str, object]], list[dict[str, object]]]:
        del args, kwargs
        return (
            {"JS445": [overwritten]},
            [
                {
                    "dataset": "JS445",
                    "event_id": 1,
                    "excluded": 0,
                }
            ],
            [{"dataset": "JS445"}],
        )

    monkeypatch.setattr(runner, "_apply_outlier_filter", fake_filter)  # type: ignore[attr-defined]

    filtered, rows, summaries = runner._screen(
        {"JS445": [source_event]},
        threshold=3.5,
        representation="native",
        target_length=None,
    )

    assert filtered["JS445"][0].is_training
    assert rows[0]["screening_representation"] == "native"
    assert summaries[0]["split_assignment_changed"] == 0


def test_parser_defaults_match_predeclared_table() -> None:
    args = runner.build_parser().parse_args([])

    assert args.tail_threshold == 0.6
    assert args.outlier_z == 3.5
    assert args.p95_percentile == 95.0
    assert args.fixed_target == 100
