"""Tests for the vendor translation layer (src/translation_layer.py).

Covers every (from_tier, to_tier) pair per vendor, idempotence, the
anchor-never-locked invariant (hypothesis), ordering rules, a state-model round
trip built from the *structured* command fields (never by parsing text),
command-injection / XML hardening, input validation and ``translate_plan`` on a
real simulation tier series.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src import translation_layer as tl
from src.translation_layer import (
    VENDORS,
    InvalidIdentifierError,
    InvalidTierError,
    OpKind,
    TierCommand,
    TranslationError,
    UnknownVendorError,
    VendorCommand,
    plan_command_counts,
    summarize_command_counts,
    tier_transitions,
    translate,
    translate_plan,
)

TIERS = range(5)
PAIRS = list(itertools.product(TIERS, TIERS))
N_GSM = tl.DEFAULT_GSM_TRX_COUNT  # default TRX list length used by translate()


# --------------------------------------------------------------------------- #
# In-test carrier state model (driven by structured fields only)
# --------------------------------------------------------------------------- #
def expected_state(tier: int) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Independent re-statement of the tier definitions (not using the module's helper)."""
    n_lte = [0, 0, 1, 2, 3][tier]
    lte = tuple(i < n_lte for i in range(3))
    if tier == 0:
        gsm = (False,) * N_GSM
    elif tier == 1:
        gsm = tuple(i == 0 for i in range(N_GSM))  # BCCH TRX only
    else:
        gsm = (True,) * N_GSM
    return lte, gsm


def apply_commands(state, cmds: list[VendorCommand]):
    """Apply commands one by one; also yields the intermediate states for ordering checks."""
    lte, gsm = list(state[0]), list(state[1])
    history = []
    for c in cmds:
        on = c.op in (OpKind.LTE_UNLOCK, OpKind.GSM_TRX_UNLOCK)
        arr = lte if c.op in (OpKind.LTE_LOCK, OpKind.LTE_UNLOCK) else gsm
        assert arr[c.index] != on, f"command is a no-op (not minimal): {c.op} idx={c.index}"
        arr[c.index] = on
        history.append((tuple(lte), tuple(gsm)))
    return (tuple(lte), tuple(gsm)), history


# --------------------------------------------------------------------------- #
# Model cross-check
# --------------------------------------------------------------------------- #
def test_tier_model_matches_reference_twin():
    from tests.reference import twin_ref

    assert tuple(twin_ref.TIER_NAMES) == tl.TIER_NAMES
    assert tuple(twin_ref.TIER_N_LTE) == tl.TIER_N_LTE


# --------------------------------------------------------------------------- #
# All pairs, all vendors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("vendor", VENDORS)
@pytest.mark.parametrize("t_from,t_to", PAIRS)
def test_every_pair_round_trip_reaches_target_state(vendor, t_from, t_to):
    cmds = translate("SITE_1", t_from, t_to, vendor)
    final, _ = apply_commands(expected_state(t_from), cmds)
    assert final == expected_state(t_to)
    # minimality: exactly the changed radios, nothing else
    lte_a, gsm_a = expected_state(t_from)
    lte_b, gsm_b = expected_state(t_to)
    changed = sum(a != b for a, b in zip(lte_a, lte_b)) + sum(a != b for a, b in zip(gsm_a, gsm_b))
    assert len(cmds) == changed
    for c in cmds:
        assert c.vendor == vendor and c.dry_run is True
        assert c.command and c.verify and c.rollback
        assert c.site_id == "SITE_1" and (c.from_tier, c.to_tier) == (t_from, t_to)


@pytest.mark.parametrize("vendor", VENDORS)
@pytest.mark.parametrize("t_from,t_to", PAIRS)
def test_rollback_restores_previous_state(vendor, t_from, t_to):
    """Each command's rollback text equals the inverse action's command text."""
    cmds = translate("SITE_1", t_from, t_to, vendor)
    for c in cmds:
        inv = translate("SITE_1", t_to, t_from, vendor)  # full reverse stream
        # the reverse stream must contain the inverse op on the same target
        import re

        norm = lambda s: re.sub(r'message-id="\d+"', 'message-id="N"', s)  # noqa: E731
        assert any(i.target == c.target and i.op is c.op.inverse and norm(i.command) == norm(c.rollback) for i in inv)


@pytest.mark.parametrize("vendor", VENDORS)
@pytest.mark.parametrize("tier", TIERS)
def test_idempotent_same_tier_is_empty(vendor, tier):
    assert translate("SITE_1", tier, tier, vendor) == []


@pytest.mark.parametrize("vendor", VENDORS)
@pytest.mark.parametrize("t_from,t_to", PAIRS)
def test_ordering_rules(vendor, t_from, t_to):
    cmds = translate("SITE_1", t_from, t_to, vendor)
    lte = [c for c in cmds if c.layer == "LTE"]
    gsm = [c for c in cmds if c.layer == "GSM"]
    if t_to < t_from:  # sleeping
        assert all(c.op in (OpKind.LTE_LOCK, OpKind.GSM_TRX_LOCK) for c in cmds)
        assert [c.index for c in lte] == sorted((c.index for c in lte), reverse=True)  # capacity high->low, anchor last
        assert [c.index for c in gsm] == sorted((c.index for c in gsm), reverse=True)  # BCCH TRX last
        # LTE is torn down before the GSM safety layer
        assert cmds == lte + gsm
    elif t_to > t_from:  # waking
        assert all(c.op in (OpKind.LTE_UNLOCK, OpKind.GSM_TRX_UNLOCK) for c in cmds)
        assert [c.index for c in lte] == sorted(c.index for c in lte)  # anchor first
        assert [c.index for c in gsm] == sorted(c.index for c in gsm)  # BCCH TRX first
        # GSM layer is up before LTE
        assert cmds == gsm + lte
    else:
        assert cmds == []
    if lte and t_from < 2 <= t_to:
        assert lte[0].index == 0  # waking from <2: anchor is first LTE carrier


def test_concrete_full_to_survival_order_huawei():
    cmds = translate("SITE_1", 4, 1, "huawei_mml")
    assert [(c.op, c.index) for c in cmds] == [
        (OpKind.LTE_LOCK, 2),
        (OpKind.LTE_LOCK, 1),
        (OpKind.LTE_LOCK, 0),
        (OpKind.GSM_TRX_LOCK, 1),
    ]
    assert cmds[0].command == "BLK CELL:LocalCellId=2;\nDEA CELL:LocalCellId=2;"
    assert cmds[0].rollback == "ACT CELL:LocalCellId=2;\nUBL CELL:LocalCellId=2;"
    assert cmds[0].statements == ("BLK CELL:LocalCellId=2;", "DEA CELL:LocalCellId=2;")
    assert cmds[0].verify.startswith("DSP CELL")


def test_concrete_wake_ericsson_text():
    cmds = translate("SITE_1", 2, 3, "ericsson_cli", cells=["A_1", "A_2", "A_3"], node="ERBS_7")
    assert len(cmds) == 1
    assert cmds[0].command == "cmedit set ERBS_7 EUtranCellFDD.(EUtranCellFDDId==A_2) administrativeState=UNLOCKED"
    assert "administrativeState=LOCKED" in cmds[0].rollback
    assert "cmedit get ERBS_7" in cmds[0].verify


def test_gsm_layer_tier1_and_tier0():
    # tier 1: LTE all off, GSM keeps only the BCCH TRX
    to_surv = translate("S1", 2, 1, "ericsson_cli")
    assert [(c.op, c.index) for c in to_surv] == [(OpKind.LTE_LOCK, 0), (OpKind.GSM_TRX_LOCK, 1)]
    assert to_surv[1].command.splitlines()[0].startswith("RXBLI:MO=")
    # tier 0: everything off, BCCH TRX is the very last action
    off = translate("S1", 4, 0, "ericsson_cli")
    assert off[-1].op is OpKind.GSM_TRX_LOCK and off[-1].index == 0
    assert sum(c.layer == "LTE" for c in off) == 3
    # waking from OFF: BCCH TRX first
    on = translate("S1", 0, 1, "ericsson_cli")
    assert [(c.op, c.index) for c in on] == [(OpKind.GSM_TRX_UNLOCK, 0)]


def test_custom_gsm_trx_list_and_single_trx():
    cmds = translate("S1", 4, 0, "huawei_mml", gsm_trx=["10", "11", "12"])
    gsm = [c for c in cmds if c.layer == "GSM"]
    assert [c.target for c in gsm] == ["12", "11", "10"]
    single = translate("S1", 1, 2, "oran_netconf", gsm_trx=["BCCH_ONLY"])
    assert [c.layer for c in single] == ["LTE"]  # min == full for a 1-TRX site


# --------------------------------------------------------------------------- #
# Property test: anchor never locked unless target tier is 0 or 1
# --------------------------------------------------------------------------- #
@settings(max_examples=150, deadline=None)
@given(
    t_from=st.integers(0, 4),
    t_to=st.integers(0, 4),
    vendor=st.sampled_from(VENDORS),
    site=st.from_regex(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,20}", fullmatch=True),
)
def test_property_anchor_never_locked_and_state_consistent(t_from, t_to, vendor, site):
    cmds = translate(site, t_from, t_to, vendor)
    for c in cmds:
        if c.op is OpKind.LTE_LOCK and c.index == 0:
            assert t_to in (0, 1)
    final, history = apply_commands(expected_state(t_from), cmds)
    assert final == expected_state(t_to)
    # anchor is never locked while a capacity carrier is still on (ordering invariant)
    for lte, _gsm in history:
        assert lte[0] or not (lte[1] or lte[2])
    # idempotent on the result
    assert translate(site, t_to, t_to, vendor) == []


@settings(max_examples=100, deadline=None)
@given(t_from=st.integers(0, 4), t_to=st.integers(0, 4), vendor=st.sampled_from(VENDORS))
def test_property_no_capacity_carrier_without_anchor_at_any_step(t_from, t_to, vendor):
    _, history = apply_commands(expected_state(t_from), translate("S", t_from, t_to, vendor))
    for lte, gsm in history:
        assert lte[0] or not (lte[1] or lte[2]), "capacity carrier on without anchor"
        # GSM safety layer: while any LTE is up, the BCCH TRX must be up
        assert gsm[0] or not any(lte)


# --------------------------------------------------------------------------- #
# Hardening: injection / XML
# --------------------------------------------------------------------------- #
BAD_IDS = [
    "SITE_1; DEA CELL:LocalCellId=0",
    "SITE_1\nDEA CELL:LocalCellId=0;",
    "SITE_1\r",
    "SITE_1\n",  # trailing newline must not slip through a '$'-style match
    'SITE_1"',
    "SITE_1'",
    "SITE<1>",
    "SITE&1",
    "SITE 1",
    "SITE_1 && rm -rf /",
    "$(reboot)",
    "`id`",
    "-flag",
    "",
    "SITE_é",
    "S" * 65,
    "SITE_1\x00",
]


@pytest.mark.parametrize("vendor", VENDORS)
@pytest.mark.parametrize("bad", BAD_IDS)
def test_bad_site_ids_rejected(vendor, bad):
    with pytest.raises(InvalidIdentifierError):
        translate(bad, 1, 4, vendor)
    with pytest.raises(InvalidIdentifierError):
        translate(bad, 2, 2, vendor)  # even when nothing would be emitted


@pytest.mark.parametrize("vendor", ["ericsson_cli", "oran_netconf"])
@pytest.mark.parametrize("bad", BAD_IDS)
def test_bad_cell_and_node_names_rejected(vendor, bad):
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, vendor, cells=["a", bad, "c"])
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, vendor, node=bad)
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, vendor, gsm_trx=[bad])


@pytest.mark.parametrize("bad", ["0; DEA CELL", "abc", "-1", "1.5", ""])
def test_huawei_ids_must_be_numeric(bad):
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, "huawei_mml", cells=["0", bad, "2"])
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, "huawei_mml", gsm_trx=[bad])


def test_empty_site_and_non_string_site():
    for bad in ("", None, 5, b"SITE_1", ["SITE_1"]):
        with pytest.raises(InvalidIdentifierError):
            translate(bad, 0, 1, "huawei_mml")


def test_cells_shape_validation():
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, "ericsson_cli", cells=["a", "b"])  # wrong count
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, "ericsson_cli", cells=["a", "a", "b"])  # duplicate
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, "ericsson_cli", cells="abc")  # a str is not a cell list
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 4, "ericsson_cli", gsm_trx=[])


@pytest.mark.parametrize("vendor", VENDORS)
def test_no_unvalidated_text_leaks_into_commands(vendor):
    cmds = translate("SITE_1", 4, 0, vendor)
    for c in cmds:
        for text in (c.command, c.verify, c.rollback):
            assert "SITE_1; " not in text
            assert "SITE_1" in text or c.target in text
    # statements are only ever internal-template lines
    assert all("\n" not in s for c in cmds for s in c.statements)


@pytest.mark.parametrize("t_from,t_to", PAIRS)
def test_netconf_payloads_are_well_formed_xml(t_from, t_to):
    for c in translate("SITE-1", t_from, t_to, "oran_netconf", cells=["C_1", "C_2", "C_3"]):
        for payload in (c.command, c.verify, c.rollback):
            root = ET.fromstring(payload)
            assert root.tag.endswith("}rpc")
            assert root.attrib["message-id"]
        ns = "{urn:ietf:params:xml:ns:netconf:base:1.0}"
        edit = ET.fromstring(c.command).find(f"{ns}edit-config")
        assert edit is not None
        assert edit.find(f"{ns}target/{ns}running") is not None
        states = [e.text for e in ET.fromstring(c.command).iter() if e.tag.endswith("administrativeState")]
        assert states == [c.target_state]
        rb = [e.text for e in ET.fromstring(c.rollback).iter() if e.tag.endswith("administrativeState")]
        assert rb == [c.op.inverse.target_state]
        assert ET.fromstring(c.verify).find(f"{ns}get") is not None


def test_netconf_payload_escapes_and_cannot_be_broken_out_of():
    # a hostile name never reaches the template ...
    with pytest.raises(InvalidIdentifierError):
        translate("S1", 0, 2, "oran_netconf", cells=["</id><x/>", "b", "c"])
    # ... and the escaper itself is the second line of defence
    assert tl._x("a<b>&'\"") == "a&lt;b&gt;&amp;'\""
    cmd = translate("S1", 0, 2, "oran_netconf")[-1]
    assert "<x/>" not in cmd.command


# --------------------------------------------------------------------------- #
# Invalid tiers / vendors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [-1, 5, 99, True, False, 1.0, 2.5, "2", None, [1], float("nan")])
@pytest.mark.parametrize("vendor", VENDORS)
def test_invalid_tiers_refused(vendor, bad):
    with pytest.raises(InvalidTierError):
        translate("S1", bad, 2, vendor)
    with pytest.raises(InvalidTierError):
        translate("S1", 2, bad, vendor)


def test_numpy_integer_tiers_accepted_numpy_bool_refused():
    import numpy as np

    assert len(translate("S1", np.int64(1), np.int64(2), "huawei_mml")) == 2  # LTE anchor + GSM TRX1
    with pytest.raises(InvalidTierError):
        translate("S1", np.bool_(True), 2, "huawei_mml")


@pytest.mark.parametrize("bad", ["", "nokia", "HUAWEI_MML", "huawei", None, 3, "huawei_mml; rm"])
def test_unknown_vendor_refused(bad):
    with pytest.raises(UnknownVendorError):
        translate("S1", 0, 4, bad)
    with pytest.raises(UnknownVendorError):
        translate("S1", 2, 2, bad)  # even for a no-op


def test_all_errors_are_translation_and_value_errors():
    for exc in (InvalidTierError, UnknownVendorError, InvalidIdentifierError):
        assert issubclass(exc, TranslationError) and issubclass(exc, ValueError)


def test_vendor_command_cannot_be_marked_live_and_is_frozen():
    c = translate("S1", 1, 2, "huawei_mml")[0]
    with pytest.raises(TranslationError):
        VendorCommand(c.vendor, c.op, c.target, c.index, c.command, c.verify, c.rollback, dry_run=False)
    with pytest.raises(Exception):
        c.command = "x"  # type: ignore[misc]


def test_tier_command_dataclass_validates():
    tc = TierCommand("S1", 1, 3, reason="safety_arbiter: heat clamp")
    assert tc.reason.startswith("safety_arbiter")
    with pytest.raises(InvalidTierError):
        TierCommand("S1", 1, 7)
    with pytest.raises(InvalidIdentifierError):
        TierCommand("S1;", 1, 2)
    assert [c.vendor for c in tl.translate_tier_command(tc, "ericsson_cli")] == ["ericsson_cli"] * 3


def test_module_is_pure():
    """No network / process-execution modules are imported by the translation layer."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(tl))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"socket", "subprocess", "requests", "urllib", "http", "ssl", "paramiko", "ncclient", "os"}


# --------------------------------------------------------------------------- #
# translate_plan / summaries
# --------------------------------------------------------------------------- #
def test_translate_plan_minimal_stream_synthetic():
    seq = [4, 4, 4, 2, 2, 1, 1, 1, 0, 0, 2, 4, 4]
    for vendor in VENDORS:
        cmds = translate_plan("SITE_1", seq, vendor)
        # replaying the stream reproduces the final tier, and the per-step tiers along the way
        state = expected_state(seq[0])
        by_step: dict[int, list[VendorCommand]] = {}
        for c in cmds:
            by_step.setdefault(c.step, []).append(c)
        for i, t in enumerate(seq[1:], start=1):
            state, _ = apply_commands(state, by_step.get(i, []))
            assert state == expected_state(t), (vendor, i)
        assert set(by_step) == {i for i in range(1, len(seq)) if seq[i] != seq[i - 1]}
        # steps are non-decreasing (stream is in time order)
        assert [c.step for c in cmds] == sorted(c.step for c in cmds)


def test_translate_plan_edge_cases():
    assert translate_plan("S1", [], "huawei_mml") == []
    assert translate_plan("S1", [3], "huawei_mml") == []
    assert translate_plan("S1", [3, 3, 3], "ericsson_cli") == []
    # explicit initial tier makes the first element a transition
    cmds = translate_plan("S1", [2], "ericsson_cli", initial_tier=4)
    assert len(cmds) == 2 and all(c.step == 0 for c in cmds)  # 4->2 locks 2 capacity carriers
    with pytest.raises(InvalidTierError):
        translate_plan("S1", [1, 2, 9], "huawei_mml")
    with pytest.raises(InvalidTierError):
        translate_plan("S1", [1, 2, 3.0], "huawei_mml")
    with pytest.raises(UnknownVendorError):
        translate_plan("S1", [1, 2], "foo")
    with pytest.raises(InvalidIdentifierError):
        translate_plan("S1;", [1, 2], "huawei_mml")
    assert tier_transitions([1, 1, 3, 3, 0]) == [(2, 1, 3), (4, 3, 0)]


def test_summarize_command_counts_and_plan_counts():
    seq = [4, 2, 4, 1]
    cmds = translate_plan("SITE_1", seq, "huawei_mml")
    summ = summarize_command_counts(cmds)
    assert set(summ) == {"huawei_mml"}
    row = summ["huawei_mml"]
    assert row["commands"] == len(cmds)
    # Huawei LTE lock/unlock are 2 statements each, GSM TRX ops 1
    assert row["statements"] == 2 * (row["lte_lock"] + row["lte_unlock"]) + row["gsm_trx_lock"] + row["gsm_trx_unlock"]
    # 4->2 locks 2 LTE, 2->4 unlocks 2 LTE, 4->1 locks 3 LTE + 1 GSM TRX
    assert row["lte_lock"] == 5 and row["lte_unlock"] == 2 and row["gsm_trx_lock"] == 1
    assert summarize_command_counts([]) == {}
    allv = plan_command_counts("SITE_1", seq)
    assert set(allv) == set(VENDORS)
    assert len({allv[v]["commands"] for v in VENDORS}) == 1  # same actions regardless of vendor
    e = allv["ericsson_cli"]
    assert e["statements"] == e["commands"] + e["gsm_trx_lock"] + e["gsm_trx_unlock"]
    assert allv["oran_netconf"]["statements"] == allv["oran_netconf"]["commands"]


def test_translate_plan_on_real_simulation_history():
    """Run the real twin and translate its enforced tier series into a command stream."""
    from src.controller import run_simulation
    from src.grid_client import BargheManClient

    res = run_simulation(
        site_id="SITE_1",
        days=2,
        wake_latency_min=5.0,
        soh=0.5,
        grid_client=BargheManClient(mode="mock"),
        return_history=True,
    )
    _metrics, hist = res
    tiers = [int(t) for t in hist["twin_tier"]]
    transitions = tier_transitions(tiers)
    for vendor in VENDORS:
        cmds = translate_plan("SITE_1", tiers, vendor)
        # one command stream, replay reproduces final tier
        final, _ = apply_commands(expected_state(tiers[0]), cmds)
        assert final == expected_state(tiers[-1])
        assert {c.step for c in cmds} <= {s for s, _, _ in transitions}
        if transitions:
            assert cmds
    # also accepts a pandas Series directly (numpy ints)
    assert translate_plan("SITE_1", hist["twin_tier"], "oran_netconf") == translate_plan("SITE_1", tiers, "oran_netconf")
    # the stream is far smaller than "one command set per step"
    assert len(translate_plan("SITE_1", tiers, "huawei_mml")) <= 5 * max(1, len(transitions))
