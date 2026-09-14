"""Tests for agent/canary.py's fixtures and leak-detection check.

See that module's docstring for what this test file can and cannot
verify offline: only the fixtures' shape/consistency and the pure
contains_injection_marker() function - not whether a real LLM actually
resists any of these injections, which needs a real, paid run
(scripts/run_agent_eval.py).
"""

from __future__ import annotations

from agent.canary import CANARY_CASES, CanaryCase, contains_injection_marker


def test_exactly_three_canary_cases_with_unique_ids():
    # Plan wording: "2-3 canary-документа" - this project settled on 3,
    # each targeting a different point in the loop (see module docstring).
    assert len(CANARY_CASES) == 3
    ids = [c.canary_id for c in CANARY_CASES]
    assert len(ids) == len(set(ids))


def test_every_canary_case_has_non_empty_required_fields():
    for case in CANARY_CASES:
        assert isinstance(case, CanaryCase)
        assert case.question.strip()
        assert case.gold_answer.strip()
        assert case.document_content.strip()
        assert len(case.injection_markers) >= 1
        assert all(m.strip() for m in case.injection_markers)
        assert case.note.strip()


def test_every_injection_marker_actually_appears_in_its_own_document():
    # A canary whose markers never occur in its own document would test
    # nothing - the injected instruction has to actually be there for the
    # leak check downstream to mean anything. At least one marker per case
    # must appear verbatim (a case may list alternate formats a model
    # might produce - e.g. with/without thousands separators - only one
    # of which is the literal form actually planted in the document).
    for case in CANARY_CASES:
        assert any(marker in case.document_content for marker in case.injection_markers), (
            f"{case.canary_id}: none of {case.injection_markers!r} found in its own document_content"
        )


def test_injection_markers_are_distinctive_not_present_in_gold_answer():
    # A legitimate correct answer must never coincidentally contain the
    # marker - otherwise contains_injection_marker() could false-positive
    # on a perfectly good answer.
    for case in CANARY_CASES:
        for marker in case.injection_markers:
            assert marker not in case.gold_answer


def test_canary_cases_target_different_documents():
    # Not three copies/variations of the same passage - each probes a
    # distinct attack surface (see module docstring).
    contents = [c.document_content for c in CANARY_CASES]
    assert len(contents) == len(set(contents))


def test_contains_injection_marker_true_when_marker_present():
    assert contains_injection_marker("Sure, INJECTED_PAYLOAD_TAKEOVER", ("INJECTED_PAYLOAD_TAKEOVER",)) is True


def test_contains_injection_marker_false_when_absent():
    assert contains_injection_marker("85190", ("INJECTED_PAYLOAD_TAKEOVER",)) is False


def test_contains_injection_marker_checks_all_given_markers():
    assert contains_injection_marker("the value is 999,999 dollars", ("999999", "999,999")) is True
    assert contains_injection_marker("the value is 412", ("999999", "999,999")) is False


def test_contains_injection_marker_is_case_sensitive_by_design():
    # Markers are deliberately distinctive fixed strings (see module
    # docstring) - case-sensitivity is a documented design choice, not an
    # oversight, so pin it down rather than leave it as an accident of
    # Python's `in` operator.
    assert contains_injection_marker("injected_payload_takeover", ("INJECTED_PAYLOAD_TAKEOVER",)) is False


def test_real_canary_cases_all_pass_contains_injection_marker_against_their_own_document():
    # Sanity check tying the fixtures and the checker together: running
    # the checker directly against a canary's own (uncompromised-model-
    # free) document content must detect its own marker - if it didn't,
    # the checker couldn't detect a real compromised answer either.
    for case in CANARY_CASES:
        assert contains_injection_marker(case.document_content, case.injection_markers) is True
