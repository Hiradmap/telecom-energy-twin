"""Unit and integration tests for BargheManClient (src/grid_client.py).

Tests:
1. Dual-mode initialization ('mock' vs 'live') and invalid mode validation.
2. SAAPA schema compliance for planned outage payloads.
3. Offline mock timetable generation, date-range filtering, and dataframe ingestion.
4. Active and upcoming blackout queries (get_upcoming_outage, is_outage_scheduled).
5. Controller integration (build_schedule_map).
6. Live mode token management, caching, auth failure (401), and network exception handling.
"""

import datetime
import pytest
import requests
import pandas as pd
from unittest.mock import MagicMock, patch

from src.grid_client import (
    BargheManClient,
    BargheManError,
    BargheManAuthError,
    BargheManAPIError,
    BargheManNetworkError,
    TAVANIR_BLOCKS,
)


# -----------------------------------------------------------------------------
# 1. Initialization & Schema Validation
# -----------------------------------------------------------------------------
def test_client_init_and_mode_validation():
    mock_client = BargheManClient(mode="mock")
    assert mock_client.mode == "mock"

    live_client = BargheManClient(mode="live", base_url="https://uiapi.saapa.ir")
    assert live_client.mode == "live"
    assert live_client.base_url == "https://uiapi.saapa.ir"

    with pytest.raises(ValueError, match="Invalid mode"):
        BargheManClient(mode="invalid_mode")


def test_mock_auth_token_caching():
    client = BargheManClient(mode="mock")
    t1 = client.authenticate()
    assert t1.startswith("mock-jwt-saapa")
    # Cached token should return identical string without recomputing
    t2 = client.authenticate()
    assert t1 == t2


def test_saapa_payload_schema_conformance():
    """Verify that returned planned outages strictly conform to the reversed SAAPA API schema."""
    client = BargheManClient(mode="mock")
    outages = client.fetch_planned_outages("SITE_1", from_date="2026-01-01", to_date="2026-01-07")

    assert len(outages) > 0
    required_keys = {
        "bill_id", "outage_id", "feeder_id", "district_code",
        "start_time", "end_time", "duration_minutes", "reason_code",
        "reason_fa", "status"
    }

    for record in outages:
        assert required_keys.issubset(record.keys()), f"Missing keys in {record}"
        assert record["bill_id"] == "SITE_1"
        assert record["duration_minutes"] == 120
        assert record["status"] == "CONFIRMED"
        assert record["reason_code"] == "LOAD_MANAGEMENT_TAVANIR"
        # Validate ISO 8601 timestamps
        s_dt = pd.to_datetime(record["start_time"])
        e_dt = pd.to_datetime(record["end_time"])
        assert e_dt > s_dt
        assert (e_dt - s_dt).total_seconds() == 7200


# -----------------------------------------------------------------------------
# 2. Mock Mode Operations & Dataframe Ingestion
# -----------------------------------------------------------------------------
def test_mock_schedule_ingestion_from_telemetry_df():
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    client = BargheManClient(mode="mock")
    client.load_mock_from_dataframe(df)

    outages = client.fetch_planned_outages("SITE_1")
    assert len(outages) > 0

    first = outages[0]
    assert first["bill_id"] == "SITE_1"
    assert first["feeder_id"] == df[df["site_id"] == "SITE_1"]["feeder_id"].iloc[0]


def test_upcoming_and_active_blackout_detection():
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    client = BargheManClient(mode="mock")
    client.load_mock_from_dataframe(df)

    outages = client.fetch_planned_outages("SITE_1")
    first_outage = outages[0]
    s_ts = pd.to_datetime(first_outage["start_time"])
    e_ts = pd.to_datetime(first_outage["end_time"])

    # 1. Query right in the middle of the blackout -> should be active
    mid_ts = s_ts + pd.Timedelta(minutes=30)
    is_active, end_val = client.is_outage_scheduled("SITE_1", mid_ts)
    assert is_active is True
    assert end_val == e_ts

    up_active = client.get_upcoming_outage("SITE_1", mid_ts)
    assert up_active is not None
    assert up_active["is_active"] is True
    assert pytest.approx(up_active["remaining_hours"], rel=1e-2) == 1.5

    # 2. Query 1 hour before the blackout -> should be upcoming
    before_ts = s_ts - pd.Timedelta(hours=1)
    is_active_b, end_val_b = client.is_outage_scheduled("SITE_1", before_ts)
    assert is_active_b is False
    assert end_val_b is None

    up_future = client.get_upcoming_outage("SITE_1", before_ts)
    assert up_future is not None
    assert up_future["is_active"] is False
    assert pytest.approx(up_future["hours_until_start"], rel=1e-2) == 1.0


def test_build_schedule_map_controller_integration():
    """Verify that build_schedule_map provides the exact mapping required by controller.py."""
    df = pd.read_csv("data/synthetic_telecom_data.csv", parse_dates=["timestamp"])
    client = BargheManClient(mode="mock")
    sched_map = client.build_schedule_map(df, site_id="SITE_1")

    assert len(sched_map) > 0
    # Pick a row that has blackout_schedule_flag == 1
    s1 = df[df["site_id"] == "SITE_1"].reset_index(drop=True)
    sched_rows = s1[s1["blackout_schedule_flag"] == 1]
    assert len(sched_rows) > 0

    sample_idx = sched_rows.index[0]
    sample_ts = sched_rows.loc[sample_idx, "timestamp"]

    # Both index and timestamp keys must exist in sched_map
    assert sample_idx in sched_map
    assert sample_ts in sched_map
    assert sched_map[sample_idx] > sample_ts


# -----------------------------------------------------------------------------
# 3. Live Mode Authentication & HTTP Resilience
# -----------------------------------------------------------------------------
def test_live_auth_token_caching_and_refresh():
    session_mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"access_token": "live-jwt-token-9988", "expires_in": 3600}
    session_mock.post.return_value = mock_resp

    client = BargheManClient(mode="live", session=session_mock, username="test_user", password="secret_pwd")
    t1 = client.authenticate()
    assert t1 == "live-jwt-token-9988"
    assert session_mock.post.call_count == 1

    # Second call should use cached token without calling post again
    t2 = client.authenticate()
    assert t2 == "live-jwt-token-9988"
    assert session_mock.post.call_count == 1

    # Force refresh should invoke post again
    t3 = client.authenticate(force_refresh=True)
    assert t3 == "live-jwt-token-9988"
    assert session_mock.post.call_count == 2


def test_live_auth_failure_raises_typed_error():
    session_mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    session_mock.post.return_value = mock_resp

    client = BargheManClient(mode="live", session=session_mock)
    with pytest.raises(BargheManAuthError):
        client.authenticate()


def test_live_fetch_network_error_handling():
    session_mock = MagicMock()
    # Mock auth success
    auth_resp = MagicMock()
    auth_resp.status_code = 200
    auth_resp.json.return_value = {"token": "jwt-token-123", "expires_in": 3600}
    session_mock.post.return_value = auth_resp

    # Mock get raising connection timeout
    session_mock.get.side_effect = requests.Timeout("Connection timed out to uiapi.saapa.ir")

    client = BargheManClient(mode="live", session=session_mock)
    with pytest.raises(BargheManNetworkError, match="Network error querying SAAPA"):
        client.fetch_planned_outages("1234567890123")


def test_live_fetch_401_triggers_token_refresh_and_retry():
    session_mock = MagicMock()
    # Auth response
    auth_resp = MagicMock()
    auth_resp.status_code = 200
    auth_resp.json.return_value = {"token": "fresh-token-456", "expires_in": 3600}
    session_mock.post.return_value = auth_resp

    # First GET returns 401 (expired token), second GET returns 200 with data
    get_401 = MagicMock()
    get_401.status_code = 401

    get_200 = MagicMock()
    get_200.status_code = 200
    get_200.json.return_value = [
        {
            "bill_id": "1234567890123",
            "outage_id": "SAAPA-TEST-1",
            "feeder_id": "F_1",
            "district_code": "THR-01",
            "start_time": "2026-06-15T11:00:00+03:30",
            "end_time": "2026-06-15T13:00:00+03:30",
            "duration_minutes": 120,
            "reason_code": "LOAD_MANAGEMENT_TAVANIR",
            "reason_fa": "مدیریت اضطراری بار",
            "status": "CONFIRMED",
        }
    ]
    session_mock.get.side_effect = [get_401, get_200]

    client = BargheManClient(mode="live", session=session_mock)
    client._token = "stale-token-111"
    client._token_expires_at = datetime.datetime.now().timestamp() + 1000

    results = client.fetch_planned_outages("1234567890123")
    assert len(results) == 1
    assert results[0]["outage_id"] == "SAAPA-TEST-1"
    # Should have refreshed token and called GET twice
    assert session_mock.post.call_count == 1
    assert session_mock.get.call_count == 2


# -----------------------------------------------------------------------------
# 4. Critical Edge Cases & Robustness
# -----------------------------------------------------------------------------
def test_outage_end_boundary_transitions_to_restored_state():
    """At the exact end timestamp of an outage (e.g. 11:00:00), grid power is restored.
    is_outage_scheduled must return False and get_upcoming_outage must not report it as active."""
    client = BargheManClient(mode="mock")
    outages = client.fetch_planned_outages("SITE_1", from_date="2026-01-01", to_date="2026-01-01")
    assert len(outages) > 0
    first = outages[0]
    e_ts = pd.to_datetime(first["end_time"])

    is_active, end_v = client.is_outage_scheduled("SITE_1", e_ts)
    assert is_active is False
    assert end_v is None

    up = client.get_upcoming_outage("SITE_1", e_ts)
    if up is not None:
        assert up["is_active"] is False


def test_date_filtering_covers_full_day_and_overlaps():
    """Date strings without time (e.g. YYYY-MM-DD) must cover the entire 24h day and ongoing overlaps."""
    client = BargheManClient(mode="mock")
    # Query single date
    day_outages = client.fetch_planned_outages("SITE_1", from_date="2026-01-01", to_date="2026-01-01")
    assert len(day_outages) >= 1
    assert any("2026-01-01" in o["start_time"] for o in day_outages)

    # Query mid-outage window [10:00, 10:30] for outage running [09:00, 11:00]
    overlap_outages = client.fetch_planned_outages(
        "SITE_1",
        from_date="2026-01-01 10:00:00+03:30",
        to_date="2026-01-01 10:30:00+03:30"
    )
    assert len(overlap_outages) >= 1


def test_unordered_outage_payload_returns_earliest_upcoming():
    """When outages are returned out of order, get_upcoming_outage must return the chronologically earliest one."""
    client = BargheManClient(mode="mock")
    client._mock_schedules["SITE_TEST"] = [
        {
            "bill_id": "SITE_TEST",
            "start_time": "2026-06-20T15:00:00+03:30",
            "end_time": "2026-06-20T17:00:00+03:30",
            "duration_minutes": 120,
        },
        {
            "bill_id": "SITE_TEST",
            "start_time": "2026-06-15T09:00:00+03:30",
            "end_time": "2026-06-15T11:00:00+03:30",
            "duration_minutes": 120,
        },
    ]

    up = client.get_upcoming_outage("SITE_TEST", "2026-06-10 12:00:00+03:30")
    assert up is not None
    assert "2026-06-15" in up["start_time"]


def test_build_schedule_map_robust_to_timezone_mismatch():
    """build_schedule_map must handle timezone-naive vs timezone-aware timestamps without TypeError."""
    client = BargheManClient(mode="live")
    # Live mode returning naive timestamps
    client.fetch_planned_outages = lambda *args, **kwargs: [
        {"start_time": "2026-01-01T09:00:00", "end_time": "2026-01-01T11:00:00"}
    ]
    # Telemetry DataFrame with tz-aware Asia/Tehran
    df_aware = pd.DataFrame({
        "site_id": ["SITE_1"] * 5,
        "timestamp": pd.date_range("2026-01-01 09:00", periods=5, freq="5min", tz="Asia/Tehran"),
        "grid_status": [0, 0, 0, 0, 0],
        "blackout_schedule_flag": [1, 1, 1, 1, 1],
    })
    sched_map = client.build_schedule_map(df_aware, "SITE_1")
    assert len(sched_map) > 0


def test_mock_client_query_unknown_site_auto_generates_and_caches():
    """Querying an uninitialized site ID in mock mode should auto-generate and cache synthetic schedule."""
    client = BargheManClient(mode="mock")
    # Initially cache is empty
    assert "SITE_UNKNOWN" not in client._mock_schedules

    outages = client.fetch_planned_outages("SITE_UNKNOWN")
    assert len(outages) > 0
    assert "SITE_UNKNOWN" in client._mock_schedules

