from gec_tagger_train.decode import apply_tags_iterative, apply_tags_once


def test_keep_returns_input_unchanged() -> None:
    out = apply_tags_once(["he", "go", "home"], ["$KEEP", "$KEEP", "$KEEP"])
    assert out == ["he", "go", "home"]


def test_replace_substitutes_token() -> None:
    out = apply_tags_once(["he", "go", "home"], ["$KEEP", "$REPLACE_goes", "$KEEP"])
    assert out == ["he", "goes", "home"]


def test_delete_removes_token() -> None:
    out = apply_tags_once(["he", "the", "goes"], ["$KEEP", "$DELETE", "$KEEP"])
    assert out == ["he", "goes"]


def test_append_inserts_after_anchor() -> None:
    out = apply_tags_once(["he", "home"], ["$APPEND_goes", "$KEEP"])
    assert out == ["he", "goes", "home"]


def test_unknown_tag_treated_as_keep() -> None:
    out = apply_tags_once(["he", "go"], ["$KEEP", "$WAT"])
    assert out == ["he", "go"]


def test_transform_case_lower() -> None:
    out = apply_tags_once(["HELLO", "World"], ["$TRANSFORM_CASE_LOWER", "$KEEP"])
    assert out == ["hello", "World"]


def test_transform_case_upper() -> None:
    out = apply_tags_once(["hello"], ["$TRANSFORM_CASE_UPPER"])
    assert out == ["HELLO"]


def test_transform_case_capital() -> None:
    out = apply_tags_once(["hello"], ["$TRANSFORM_CASE_CAPITAL"])
    assert out == ["Hello"]


def test_transform_verb_falls_through_as_keep() -> None:
    out = apply_tags_once(["go"], ["$TRANSFORM_VERB_VB_VBZ"])
    assert out == ["go"]


def test_iterative_stops_on_noop_pass() -> None:
    calls = []

    def tagger(toks: list[str]) -> list[str]:
        calls.append(list(toks))
        return ["$KEEP"] * len(toks)

    out = apply_tags_iterative(["he", "is", "ok"], tagger, max_iter=3)
    assert out == ["he", "is", "ok"]
    assert len(calls) == 1


def test_iterative_chains_two_passes() -> None:
    sequence = iter([
        ["$KEEP", "$REPLACE_goes", "$KEEP"],     # pass 1: he go home -> he goes home
        ["$TRANSFORM_CASE_CAPITAL", "$KEEP", "$KEEP"],  # pass 2: capitalize first
        ["$KEEP", "$KEEP", "$KEEP"],             # pass 3: no-op, stop
    ])

    def tagger(_toks: list[str]) -> list[str]:
        return next(sequence)

    out = apply_tags_iterative(["he", "go", "home"], tagger, max_iter=5)
    assert out == ["He", "goes", "home"]


def test_iterative_respects_max_iter() -> None:
    def tagger(toks: list[str]) -> list[str]:
        # Always proposes one replacement; iterative would loop forever.
        if toks and toks[0] != "DONE":
            return ["$REPLACE_DONE"] + ["$KEEP"] * (len(toks) - 1)
        return ["$KEEP"] * len(toks)

    out = apply_tags_iterative(["a", "b"], tagger, max_iter=2)
    assert out == ["DONE", "b"]
