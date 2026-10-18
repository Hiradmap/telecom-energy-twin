"""Vendor translation layer (src/translation_layer.py).

Converts the vendor-neutral *service tier* command chosen by the Safety Arbiter
into vendor-specific radio on/off commands (Production_Roadmap.md §2):

    tier 0 OFF       : GSM off, LTE off
    tier 1 SURVIVAL  : GSM minimum (BCCH TRX only), LTE off
    tier 2 BASIC     : GSM full, LTE anchor carrier (index 0)
    tier 3 ANCHOR+1  : GSM full, anchor + 1 capacity carrier
    tier 4 FULL      : GSM full, anchor + 2 capacity carriers (3 LTE carriers)

Supported vendors: ``huawei_mml``, ``ericsson_cli``, ``oran_netconf``.
The last one is the "legacy layer removed" end state: a 3GPP-NRM style O1 NETCONF
payload that a native O-RAN rApp / SMO could push directly.

!!! ILLUSTRATIVE -- NOT FIELD-VALIDATED !!!
    Every command template in this file is a *dry-run illustration* assembled from
    public secondary material. Syntax, parameter names, MO/MML object names and
    YANG namespaces MUST be validated against the operator's own MOM / MML /
    YANG release (and the target OSS: U2000/MAE, ENM, SMO) before any field use.
    Places where the syntax is uncertain are flagged with ``UNCERTAIN`` comments.

Safety posture
    * Pure: no network, no subprocess, nothing is ever executed. Every
      ``VendorCommand`` is dry-run output (``dry_run`` is always True). A real
      deployment would hand these through the operator's OSS/ENM/MAE northbound
      API (with change approval / rollback) -- never through a raw shell.
    * Tiers must be real ints in 0..4 (bool, float, str, out-of-range -> error).
    * Vendor must be one of the known names; site id / cell / TRX / node names must
      match ``[A-Za-z0-9_][A-Za-z0-9_-]{0,63}`` (command-injection hardening: no
      spaces, ``;``, quotes, ``<``, newlines ...). Values are additionally
      XML-escaped inside NETCONF payloads, and each payload is parsed back with
      ``xml.etree`` before it is returned.
    * The anchor LTE carrier (index 0) is only ever locked when the target tier is
      0 or 1.

Sources consulted (secondary: forum/training material + standards; none replaces
the vendor's own command reference):
    - telecomhall.net forum threads on Huawei eRAN MML ``BLK/UBL/DEA/ACT CELL``
      and ``DSP CELL`` / ``LST CELL``.
    - Huawei BSC6900 GSM MML material (Scribd/Telecomhall): ``BLK/UBL GTRX``.
    - Ericsson ENM CLI ``cmedit set/get`` examples for
      ``EUtranCellFDD ... administrativeState=LOCKED|UNLOCKED``.
    - Ericsson BSC/RBS MML (WinFIOL) ``RXBLI / RXESE / RXESI / RXBLE / RXMSP``
      block-deblock sequence for ``RXOTRX`` objects.
    - 3GPP TS 28.541 / 28.658 (NRM), TS 28.625 / ITU-T X.731 (administrativeState =
      LOCKED | UNLOCKED | SHUTTINGDOWN); 3GPP SA5 YANG namespaces
      ``urn:3gpp:sa5:_3gpp-common-managed-element`` (as published in O-RAN SC
      O1 examples) and RFC 6241 NETCONF ``<edit-config>``.
"""

from __future__ import annotations

import numbers
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

__all__ = [
    "VENDORS",
    "TIER_NAMES",
    "TIER_N_LTE",
    "MAX_LTE_CARRIERS",
    "TranslationError",
    "InvalidTierError",
    "UnknownVendorError",
    "InvalidIdentifierError",
    "OpKind",
    "TierCommand",
    "VendorCommand",
    "translate",
    "translate_tier_command",
    "translate_plan",
    "tier_transitions",
    "target_radio_state",
    "summarize_command_counts",
    "plan_command_counts",
]

# --------------------------------------------------------------------------- #
# Tier model (mirrors tests/reference/twin_ref.py TIER_NAMES / TIER_N_LTE)
# --------------------------------------------------------------------------- #
TIER_NAMES: tuple[str, ...] = ("OFF", "SURVIVAL", "BASIC", "ANCHOR+1", "FULL")
TIER_N_LTE: tuple[int, ...] = (0, 0, 1, 2, 3)
MAX_LTE_CARRIERS = 3  # carrier index 0 = anchor, 1..2 = capacity carriers
MAX_TIER = len(TIER_NAMES) - 1

# GSM layer level per tier. Assumption (documented, not from the twin): tier 1 is
# the "GSM minimum" config (BCCH TRX only); tiers >= 2 run all configured TRXs.
_GSM_OFF, _GSM_MIN, _GSM_FULL = 0, 1, 2
_GSM_LEVEL_BY_TIER: tuple[int, ...] = (_GSM_OFF, _GSM_MIN, _GSM_FULL, _GSM_FULL, _GSM_FULL)

VENDORS: tuple[str, ...] = ("huawei_mml", "ericsson_cli", "oran_netconf")

DEFAULT_GSM_TRX_COUNT = 2

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TranslationError(ValueError):
    """Base class: the translation request was refused (nothing was generated)."""


class InvalidTierError(TranslationError):
    """Tier is not an int in 0..4 (bool / float / str / out of range)."""


class UnknownVendorError(TranslationError):
    """Vendor name is not one of :data:`VENDORS`."""


class InvalidIdentifierError(TranslationError):
    """Site id / cell / TRX / node name failed the strict whitelist regex."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
# fullmatch (not match/$) so a trailing newline can never slip through.
_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,63}")
_NUM_RE = re.compile(r"[0-9]{1,9}")


def _check_identifier(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise InvalidIdentifierError(f"{what} must be a str, got {type(value).__name__}")
    if _ID_RE.fullmatch(value) is None:
        raise InvalidIdentifierError(
            f"{what} {value!r} rejected: only letters, digits, '_' and '-' are allowed "
            f"(1-64 chars, must not start with '-')"
        )
    return value


def _check_tier(value: object, what: str = "tier") -> int:
    # bool is an int subclass: True/False must NOT be accepted as tier 1/0.
    # numbers.Integral also accepts numpy ints coming out of pandas Series.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidTierError(f"{what} must be an int in 0..{MAX_TIER}, got {value!r} ({type(value).__name__})")
    tier = int(value)
    if not 0 <= tier <= MAX_TIER:
        raise InvalidTierError(f"{what} {tier} out of range 0..{MAX_TIER}")
    return tier


def _check_vendor(vendor: object) -> str:
    if not isinstance(vendor, str) or vendor not in VENDORS:
        raise UnknownVendorError(f"unknown vendor {vendor!r}; expected one of {VENDORS}")
    return vendor


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


class OpKind(str, Enum):
    """Kind of radio-state change a single :class:`VendorCommand` performs."""

    LTE_LOCK = "lte_lock"
    LTE_UNLOCK = "lte_unlock"
    GSM_TRX_LOCK = "gsm_trx_lock"
    GSM_TRX_UNLOCK = "gsm_trx_unlock"

    @property
    def layer(self) -> str:
        return "LTE" if self in (OpKind.LTE_LOCK, OpKind.LTE_UNLOCK) else "GSM"

    @property
    def target_state(self) -> str:
        return "LOCKED" if self in (OpKind.LTE_LOCK, OpKind.GSM_TRX_LOCK) else "UNLOCKED"

    @property
    def inverse(self) -> "OpKind":
        return {
            OpKind.LTE_LOCK: OpKind.LTE_UNLOCK,
            OpKind.LTE_UNLOCK: OpKind.LTE_LOCK,
            OpKind.GSM_TRX_LOCK: OpKind.GSM_TRX_UNLOCK,
            OpKind.GSM_TRX_UNLOCK: OpKind.GSM_TRX_LOCK,
        }[self]


@dataclass(frozen=True)
class TierCommand:
    """Vendor-neutral command: move ``site_id`` from ``from_tier`` to ``to_tier``."""

    site_id: str
    from_tier: int
    to_tier: int
    reason: str = ""  # provenance, e.g. "safety_arbiter: UB clamp (thermal)"

    def __post_init__(self) -> None:
        _check_identifier(self.site_id, "site_id")
        _check_tier(self.from_tier, "from_tier")
        _check_tier(self.to_tier, "to_tier")
        if not isinstance(self.reason, str):
            raise TranslationError("reason must be a str")


@dataclass(frozen=True)
class VendorCommand:
    """One ordered, structured, dry-run action for a single cell / TRX.

    ``command`` may hold several statements separated by ``\\n`` (e.g. Huawei
    BLK + DEA); see :attr:`statements`. ``index`` is the LTE carrier index
    (0 = anchor) or the GSM TRX index (0 = BCCH TRX). Nothing here is ever executed.
    """

    vendor: str
    op: OpKind
    target: str  # cell / TRX identifier as used inside the command text
    index: int
    command: str
    verify: str  # read-back command
    rollback: str  # command restoring the previous state of this target
    site_id: str = ""
    from_tier: int | None = None
    to_tier: int | None = None
    step: int | None = None  # position in the tier series (translate_plan)
    dry_run: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.dry_run is not True:
            raise TranslationError("this layer only produces dry-run commands")

    @property
    def layer(self) -> str:
        return self.op.layer

    @property
    def target_state(self) -> str:
        return self.op.target_state

    @property
    def statements(self) -> tuple[str, ...]:
        return tuple(s for s in self.command.split("\n") if s)

    def as_text(self) -> str:
        return f"{self.command}\n! verify:   {self.verify}\n! rollback: " + self.rollback.replace("\n", "\n!           ")


# --------------------------------------------------------------------------- #
# Tier -> desired radio state / diff
# --------------------------------------------------------------------------- #


def target_radio_state(tier: int, n_gsm_trx: int = DEFAULT_GSM_TRX_COUNT) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Return ``(lte_unlocked[carrier], gsm_unlocked[trx])`` for ``tier``."""
    tier = _check_tier(tier)
    n_on = TIER_N_LTE[tier]
    lte = tuple(i < n_on for i in range(MAX_LTE_CARRIERS))
    return lte, _gsm_state(tier, n_gsm_trx)


def _gsm_state(tier: int, n_trx: int) -> tuple[bool, ...]:
    level = _GSM_LEVEL_BY_TIER[tier]
    if level == _GSM_OFF:
        return (False,) * n_trx
    if level == _GSM_MIN:
        return tuple(i == 0 for i in range(n_trx))  # BCCH TRX only
    return (True,) * n_trx


def _plan_steps(from_tier: int, to_tier: int, n_trx: int) -> list[tuple[OpKind, int]]:
    """Minimal ordered (op, index) list moving ``from_tier`` -> ``to_tier``.

    Sleeping (locks):  LTE capacity carriers highest index first, anchor last,
                       then GSM extra TRXs highest first, BCCH TRX very last.
    Waking (unlocks):  GSM BCCH TRX first, then other GSM TRXs, then LTE anchor
                       first, then capacity carriers ascending.
    Both LTE and GSM levels are monotone in the tier, so a transition is either
    all-locks or all-unlocks.
    """
    if from_tier == to_tier:
        return []
    lte_from, lte_to = TIER_N_LTE[from_tier], TIER_N_LTE[to_tier]
    gsm_from, gsm_to = _gsm_state(from_tier, n_trx), _gsm_state(to_tier, n_trx)

    locks: list[tuple[OpKind, int]] = [(OpKind.LTE_LOCK, i) for i in range(lte_from - 1, lte_to - 1, -1)]
    locks += [(OpKind.GSM_TRX_LOCK, i) for i in range(n_trx - 1, -1, -1) if gsm_from[i] and not gsm_to[i]]

    unlocks: list[tuple[OpKind, int]] = [(OpKind.GSM_TRX_UNLOCK, i) for i in range(n_trx) if gsm_to[i] and not gsm_from[i]]
    unlocks += [(OpKind.LTE_UNLOCK, i) for i in range(lte_from, lte_to)]

    return locks + unlocks


# --------------------------------------------------------------------------- #
# Vendor renderers: (op, target, node, seq) -> action text ; verify text
# --------------------------------------------------------------------------- #


# ---- Huawei MML (U2000 / MAE-Access style) -------------------------------- #
# LTE (eRAN): cell addressed by LocalCellId, so `cells` entries must be digits.
# Lock = block then deactivate; unlock = activate then unblock (sequence from
# forum material). UNCERTAIN: exact BLK CELL parameters (e.g. a block-type /
# timer parameter) and whether DEA CELL is appropriate vs. an ES-specific MO
# differ per eRAN release.
# GSM (BSC6900): TRX addressed by TRXID. UNCERTAIN: BLK/UBL GTRX naming and
# parameter set differ across GBSS releases; verify with LST GTRX / DSP GTRX.
def _huawei_action(op: OpKind, target: str, node: str, seq: int) -> str:
    if op is OpKind.LTE_LOCK:
        return f"BLK CELL:LocalCellId={target};\nDEA CELL:LocalCellId={target};"
    if op is OpKind.LTE_UNLOCK:
        return f"ACT CELL:LocalCellId={target};\nUBL CELL:LocalCellId={target};"
    if op is OpKind.GSM_TRX_LOCK:
        return f"BLK GTRX:TRXID={target};"
    return f"UBL GTRX:TRXID={target};"


def _huawei_verify(op: OpKind, target: str, node: str, seq: int) -> str:
    if op.layer == "LTE":
        return f"DSP CELL:LocalCellId={target};"
    return f"LST GTRX:TRXID={target};"  # UNCERTAIN: DSP GTRX may be the runtime-state variant


# ---- Ericsson (ENM cmedit for LTE; BSC/RBS MML for GSM) ------------------- #
# LTE: `cmedit set <node> EUtranCellFDD.(EUtranCellFDDId==<cell>) administrativeState=...`
# UNCERTAIN: exact filter syntax / need for `--force` depends on ENM release;
# the read-back attribute selection syntax below is the least certain part.
# GSM: legacy MML (WinFIOL/APG, not ENM cmedit): block -> out-of-service on lock,
# in-service -> deblock on unlock, targeting an RXOTRX managed object
# (e.g. MO=RXOTRX-45-2). UNCERTAIN: MO naming (we use the validated target text
# verbatim as the MO string) and whether SUBORD/FORCE are wanted per site.
def _ericsson_action(op: OpKind, target: str, node: str, seq: int) -> str:
    if op.layer == "LTE":
        return f"cmedit set {node} EUtranCellFDD.(EUtranCellFDDId=={target}) administrativeState={op.target_state}"
    if op is OpKind.GSM_TRX_LOCK:
        return f"RXBLI:MO={target},SUBORD;\nRXESE:MO={target},SUBORD;"
    return f"RXESI:MO={target},SUBORD;\nRXBLE:MO={target},SUBORD;"


def _ericsson_verify(op: OpKind, target: str, node: str, seq: int) -> str:
    if op.layer == "LTE":
        return f"cmedit get {node} EUtranCellFDD.(EUtranCellFDDId=={target}).(administrativeState,operationalState)"
    return f"RXMSP:MO={target},SUBORD;"


# ---- O-RAN O1 NETCONF (3GPP NRM style) ------------------------------------ #
# administrativeState values LOCKED/UNLOCKED follow TS 28.625 / X.731.
# UNCERTAIN / PLACEHOLDER: the YANG module namespaces for the LTE NRM
# (ENBFunction / EUtranCellFDD) are extrapolated by analogy with the published
# NR modules (_3gpp-nr-nrm-*); only the ManagedElement namespace is taken from
# published O1 examples. GSM has no standard O1/YANG path that I could verify,
# so the GsmTrx element and its namespace are purely illustrative placeholders.
_NC_NS = "urn:ietf:params:xml:ns:netconf:base:1.0"
_NS_ME = "urn:3gpp:sa5:_3gpp-common-managed-element"
_NS_ENB = "urn:3gpp:sa5:_3gpp-lte-nrm-enbfunction"  # UNCERTAIN (extrapolated)
_NS_CELL = "urn:3gpp:sa5:_3gpp-lte-nrm-eutrancellfdd"  # UNCERTAIN (extrapolated)
_NS_GSM = "urn:example:illustrative:gsm-trx"  # PLACEHOLDER (no verified standard)


def _x(value: str) -> str:
    return _xml_escape(value)  # defence in depth; values already regex-validated


def _oran_body(op: OpKind, target: str, node: str, inner: str) -> str:
    if op.layer == "LTE":
        return (
            f'<ManagedElement xmlns={_xml_quoteattr(_NS_ME)}><id>{_x(node)}</id>'
            f'<ENBFunction xmlns={_xml_quoteattr(_NS_ENB)}><id>1</id>'
            f'<EUtranCellFDD xmlns={_xml_quoteattr(_NS_CELL)}><id>{_x(target)}</id>{inner}</EUtranCellFDD>'
            f"</ENBFunction></ManagedElement>"
        )
    return (
        f'<ManagedElement xmlns={_xml_quoteattr(_NS_ME)}><id>{_x(node)}</id>'
        f'<GsmTrx xmlns={_xml_quoteattr(_NS_GSM)}><id>{_x(target)}</id>{inner}</GsmTrx></ManagedElement>'
    )


def _oran_action(op: OpKind, target: str, node: str, seq: int) -> str:
    inner = f"<attributes><administrativeState>{op.target_state}</administrativeState></attributes>"
    return (
        f'<rpc xmlns={_xml_quoteattr(_NC_NS)} message-id={_xml_quoteattr(str(seq))}>'
        f"<edit-config><target><running/></target><config>{_oran_body(op, target, node, inner)}</config></edit-config></rpc>"
    )


def _oran_verify(op: OpKind, target: str, node: str, seq: int) -> str:
    inner = "<attributes><administrativeState/><operationalState/></attributes>"
    return (
        f'<rpc xmlns={_xml_quoteattr(_NC_NS)} message-id={_xml_quoteattr(str(seq))}>'
        f'<get><filter type="subtree">{_oran_body(op, target, node, inner)}</filter></get></rpc>'
    )


_Renderer = Callable[[OpKind, str, str, int], str]
_RENDERERS: dict[str, tuple[_Renderer, _Renderer]] = {
    "huawei_mml": (_huawei_action, _huawei_verify),
    "ericsson_cli": (_ericsson_action, _ericsson_verify),
    "oran_netconf": (_oran_action, _oran_verify),
}


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #


def _resolve_names(
    site_id: str,
    vendor: str,
    cells: Sequence[str] | None,
    gsm_trx: Sequence[str] | None,
    node: str | None,
) -> tuple[list[str], list[str], str]:
    # Huawei MML addresses cells/TRXs by numeric ids; others by name/MO string.
    if vendor == "huawei_mml":
        lte_default = [str(i) for i in range(MAX_LTE_CARRIERS)]
        gsm_default = [str(i) for i in range(DEFAULT_GSM_TRX_COUNT)]
    elif vendor == "ericsson_cli":
        lte_default = [f"{site_id}_LTE{i}" for i in range(MAX_LTE_CARRIERS)]
        gsm_default = [f"RXOTRX-{site_id}-{i}" for i in range(DEFAULT_GSM_TRX_COUNT)]
    else:
        lte_default = [f"{site_id}_LTE{i}" for i in range(MAX_LTE_CARRIERS)]
        gsm_default = [f"{site_id}_GSM{i}" for i in range(DEFAULT_GSM_TRX_COUNT)]

    def _clean(seq: Sequence[str] | None, default: list[str], what: str) -> list[str]:
        if seq is None:
            return list(default)
        if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
            raise InvalidIdentifierError(f"{what} must be a sequence of names, got {type(seq).__name__}")
        names = [_check_identifier(n, f"{what} name") for n in seq]
        if vendor == "huawei_mml":
            for n in names:
                if _NUM_RE.fullmatch(n) is None:
                    raise InvalidIdentifierError(f"huawei_mml {what} id {n!r} must be numeric (LocalCellId / TRXID)")
        if len(set(names)) != len(names):
            raise InvalidIdentifierError(f"{what} names must be unique: {names}")
        return names

    lte = _clean(cells, lte_default, "cells")
    if len(lte) != MAX_LTE_CARRIERS:
        raise InvalidIdentifierError(
            f"cells must list exactly {MAX_LTE_CARRIERS} LTE cells ordered by carrier index (0 = anchor), got {len(lte)}"
        )
    trx = _clean(gsm_trx, gsm_default, "gsm_trx")
    if not trx:
        raise InvalidIdentifierError("gsm_trx must contain at least one TRX (index 0 = BCCH TRX)")
    node_name = site_id if node is None else _check_identifier(node, "node")
    return lte, trx, node_name


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def translate_tier_command(
    cmd: TierCommand,
    vendor: str,
    cells: Sequence[str] | None = None,
    gsm_trx: Sequence[str] | None = None,
    node: str | None = None,
) -> list[VendorCommand]:
    """Translate a :class:`TierCommand` into an ordered list of dry-run commands."""
    vendor = _check_vendor(vendor)
    lte, trx, node_name = _resolve_names(cmd.site_id, vendor, cells, gsm_trx, node)
    action, verify = _RENDERERS[vendor]
    out: list[VendorCommand] = []
    for seq, (op, idx) in enumerate(_plan_steps(cmd.from_tier, cmd.to_tier, len(trx)), start=1):
        target = lte[idx] if op.layer == "LTE" else trx[idx]
        vc = VendorCommand(
            vendor=vendor,
            op=op,
            target=target,
            index=idx,
            command=action(op, target, node_name, seq),
            verify=verify(op, target, node_name, seq),
            rollback=action(op.inverse, target, node_name, seq),
            site_id=cmd.site_id,
            from_tier=cmd.from_tier,
            to_tier=cmd.to_tier,
        )
        if vendor == "oran_netconf":
            for payload in (vc.command, vc.verify, vc.rollback):
                try:
                    ET.fromstring(payload)
                except ET.ParseError as exc:  # pragma: no cover - template bug guard
                    raise TranslationError(f"generated NETCONF payload is not well-formed: {exc}") from exc
        out.append(vc)
    return out


def translate(
    site_id: str,
    from_tier: int,
    to_tier: int,
    vendor: str,
    cells: Sequence[str] | None = None,
    gsm_trx: Sequence[str] | None = None,
    node: str | None = None,
    reason: str = "",
) -> list[VendorCommand]:
    """Translate one tier transition for ``site_id`` into vendor commands (dry run).

    ``cells``: exactly 3 LTE cell identifiers ordered by carrier index (0 = anchor);
    ``gsm_trx``: GSM TRX identifiers (index 0 = BCCH TRX); ``node``: ENM/NETCONF
    managed-element name (defaults to ``site_id``). Same tier -> ``[]``.
    """
    return translate_tier_command(TierCommand(site_id, from_tier, to_tier, reason), vendor, cells, gsm_trx, node)


def tier_transitions(tier_sequence: Iterable[object], initial_tier: int | None = None) -> list[tuple[int, int, int]]:
    """Return ``(step, from_tier, to_tier)`` for each change in ``tier_sequence``.

    ``step`` is the index in the sequence where the new tier first appears. The first
    element is taken as the already-applied state unless ``initial_tier`` is given.
    All tiers are validated, including those that do not change.
    """
    tiers = [_check_tier(t, "tier_sequence element") for t in tier_sequence]
    prev = None if initial_tier is None else _check_tier(initial_tier, "initial_tier")
    out: list[tuple[int, int, int]] = []
    for i, t in enumerate(tiers):
        if prev is not None and t != prev:
            out.append((i, prev, t))
        prev = t
    return out


def translate_plan(
    site_id: str,
    tier_sequence: Iterable[object],
    vendor: str,
    cells: Sequence[str] | None = None,
    gsm_trx: Sequence[str] | None = None,
    node: str | None = None,
    initial_tier: int | None = None,
) -> list[VendorCommand]:
    """Translate a whole tier history (e.g. the ``twin_tier`` column) to the minimal
    command stream: only actual tier changes emit commands; repeats emit nothing.

    Each returned command carries ``step`` = index in ``tier_sequence``.
    """
    _check_identifier(site_id, "site_id")
    _check_vendor(vendor)
    out: list[VendorCommand] = []
    for step, t_from, t_to in tier_transitions(tier_sequence, initial_tier):
        cmds = translate(site_id, t_from, t_to, vendor, cells, gsm_trx, node, reason=f"plan step {step}")
        out.extend(replace(c, step=step) for c in cmds)
    return out


def summarize_command_counts(commands: Iterable[VendorCommand]) -> dict[str, dict[str, int]]:
    """Count commands per vendor: ``{vendor: {"commands", "statements", <op kinds...>}}``."""
    summary: dict[str, dict[str, int]] = {}
    for c in commands:
        row = summary.setdefault(c.vendor, {"commands": 0, "statements": 0, **{op.value: 0 for op in OpKind}})
        row["commands"] += 1
        row["statements"] += len(c.statements)
        row[c.op.value] += 1
    return summary


def plan_command_counts(site_id: str, tier_sequence: Iterable[object]) -> dict[str, dict[str, int]]:
    """Command counts for the same tier history under every supported vendor."""
    tiers = list(tier_sequence)
    summary: dict[str, dict[str, int]] = {}
    for v in VENDORS:
        counts = summarize_command_counts(translate_plan(site_id, tiers, v))
        summary[v] = counts.get(v, {"commands": 0, "statements": 0, **{op.value: 0 for op in OpKind}})
    return summary
