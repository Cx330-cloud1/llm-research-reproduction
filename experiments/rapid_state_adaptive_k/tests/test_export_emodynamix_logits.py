import math

import pytest

from scripts.export_emodynamix_logits import (
    context_signature,
    join_preprocessed,
    verified_label_order,
)


MODEL_CONTEXT = {
    "dialogue_history": "<START> </s> hello",
    "strategy_history": "[-1, -1]",
    "speaker_turn": "None seeker",
}


def test_context_signature_uses_all_three_model_inputs():
    original = context_signature(MODEL_CONTEXT)

    replacements = {
        "dialogue_history": "<START> </s> changed",
        "strategy_history": "[-1, 2]",
        "speaker_turn": "seeker supporter",
    }

    for field, replacement in replacements.items():
        changed = {
            **MODEL_CONTEXT,
            field: replacement,
        }

        assert context_signature(changed) != original


def test_context_signature_treats_nan_speaker_as_none():
    manifest_context = {
        "dialogue_history": "<START>",
        "strategy_history": "[-1]",
        "speaker_turn": "None",
    }

    preprocessed_context = {
        "dialogue_history": "<START>",
        "strategy_history": "[-1]",
        "speaker_turn": math.nan,
    }

    assert (
        context_signature(manifest_context)
        == context_signature(preprocessed_context)
    )


def test_join_uses_all_three_model_inputs():
    base_rows = [
        {
            "sample_id": "s1",
            "model_context": MODEL_CONTEXT,
        }
    ]

    preprocessed_rows = [
        {
            **MODEL_CONTEXT,
            "parsed_dialogue": [],
            "erc_logits": [[0.0] * 7],
        }
    ]

    joined = join_preprocessed(
        base_rows,
        preprocessed_rows,
    )

    assert len(joined) == 1
    assert joined[0][0]["sample_id"] == "s1"
    assert joined[0][1]["parsed_dialogue"] == []
    assert joined[0][1]["erc_logits"] == [[0.0] * 7]


def test_identical_duplicate_preprocessed_rows_are_allowed():
    base_rows = [
        {
            "sample_id": "s1",
            "model_context": MODEL_CONTEXT,
        }
    ]

    row = {
        **MODEL_CONTEXT,
        "parsed_dialogue": [],
        "erc_logits": [[0.0] * 7],
    }

    joined = join_preprocessed(
        base_rows,
        [
            dict(row),
            dict(row),
        ],
    )

    assert len(joined) == 1
    assert joined[0][0]["sample_id"] == "s1"


def test_ambiguous_duplicate_preprocessed_signature_fails_closed():
    base_rows = [
        {
            "sample_id": "s1",
            "model_context": MODEL_CONTEXT,
        }
    ]

    preprocessed_rows = [
        {
            **MODEL_CONTEXT,
            "parsed_dialogue": [],
            "erc_logits": [[0.0] * 7],
        },
        {
            **MODEL_CONTEXT,
            "parsed_dialogue": [{"different": True}],
            "erc_logits": [[1.0] * 7],
        },
    ]

    with pytest.raises(
        ValueError,
        match="ambiguous preprocessed signature",
    ):
        join_preprocessed(
            base_rows,
            preprocessed_rows,
        )


def test_irrelevant_duplicate_preprocessed_signature_is_ignored():
    base_rows = [
        {
            "sample_id": "s1",
            "model_context": MODEL_CONTEXT,
        }
    ]

    irrelevant_context = {
        "dialogue_history": "<START> </s> irrelevant",
        "strategy_history": "[-1, -1]",
        "speaker_turn": "None seeker",
    }

    preprocessed_rows = [
        {
            **MODEL_CONTEXT,
            "parsed_dialogue": [],
            "erc_logits": [[0.0] * 7],
        },
        {
            **irrelevant_context,
        },
        {
            **irrelevant_context,
        },
    ]

    joined = join_preprocessed(
        base_rows,
        preprocessed_rows,
    )

    assert len(joined) == 1
    assert joined[0][0]["sample_id"] == "s1"


def test_duplicate_base_signature_is_allowed():
    base_rows = [
        {
            "sample_id": "s1",
            "model_context": MODEL_CONTEXT,
        },
        {
            "sample_id": "s2",
            "model_context": MODEL_CONTEXT,
        },
    ]

    preprocessed_rows = [
        {
            **MODEL_CONTEXT,
            "parsed_dialogue": [],
            "erc_logits": [[0.0] * 7],
        }
    ]

    joined = join_preprocessed(
        base_rows,
        preprocessed_rows,
    )

    assert len(joined) == 2
    assert [
        pair[0]["sample_id"]
        for pair in joined
    ] == ["s1", "s2"]

    assert joined[0][1] is joined[1][1]


def test_nan_preprocessed_row_can_serve_none_base_rows():
    base_context = {
        "dialogue_history": "<START>",
        "strategy_history": "[-1]",
        "speaker_turn": "None",
    }

    base_rows = [
        {
            "sample_id": "s1",
            "model_context": base_context,
        },
        {
            "sample_id": "s2",
            "model_context": base_context,
        },
    ]

    preprocessed_row = {
        "dialogue_history": "<START>",
        "strategy_history": "[-1]",
        "speaker_turn": math.nan,
        "parsed_dialogue": [],
        "erc_logits": [[0.0] * 7],
    }

    joined = join_preprocessed(
        base_rows,
        [
            dict(preprocessed_row),
            dict(preprocessed_row),
        ],
    )

    assert len(joined) == 2
    assert [
        pair[0]["sample_id"]
        for pair in joined
    ] == ["s1", "s2"]


def test_missing_preprocessed_signature_fails_closed():
    base_rows = [
        {
            "sample_id": "s1",
            "model_context": MODEL_CONTEXT,
        }
    ]

    with pytest.raises(
        ValueError,
        match="missing preprocessed signature",
    ):
        join_preprocessed(
            base_rows,
            [],
        )


def test_verified_label_order_uses_numeric_ids():
    strategy2id = {
        "Information": 1,
        "Question": 0,
        "Reflection": 2,
    }

    expected = [
        "Question",
        "Information",
        "Reflection",
    ]

    assert (
        verified_label_order(
            strategy2id,
            expected,
        )
        == expected
    )


def test_verified_label_order_rejects_mismatch():
    strategy2id = {
        "Information": 1,
        "Question": 0,
        "Reflection": 2,
    }

    expected = [
        "Information",
        "Question",
        "Reflection",
    ]

    with pytest.raises(
        ValueError,
        match="label order mismatch",
    ):
        verified_label_order(
            strategy2id,
            expected,
        )


def test_smoke20_collection_selects_only_smoke20():
    from scripts.export_emodynamix_logits import (
        selected_collections,
    )

    assert selected_collections("smoke20") == [
        "smoke20"
    ]
