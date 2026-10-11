"""Dual-Mode Bargh-e Man Adapter (src/grid_client.py).

Conforms to the reversed Iranian SAAPA (uiapi.saapa.ir) schema for Tavanir planned power rationing.
Supports:
  1. mode="mock": Default offline simulation returning realistic 2-hour Tavanir timetable payloads.
  2. mode="live": Production Iranian deployment with authentication, token caching, endpoint polling,
                  and fault-tolerant retry/fallback mechanics.
"""

from __future__ import annotations
import datetime
import time
import requests
import pandas as pd
from typing import Any


class BargheManError(Exception):
    """Base exception for Bargh-e Man / SAAPA client."""
    pass


class BargheManAuthError(BargheManError):
    """Authentication or token expiry failure."""
    pass


class BargheManAPIError(BargheManError):
    """Non-200 API response or unparseable payload."""
    pass


class BargheManNetworkError(BargheManError):
    """Network connection, DNS, or request timeout failure."""
    pass


# Canonical 2-hour Tavanir planned rationing blocks
TAVANIR_BLOCKS = [
    ("09:00", "11:00"),
    ("11:00", "13:00"),
    ("13:00", "15:00"),
    ("15:00", "17:00"),
    ("17:00", "19:00"),
    ("19:00", "21:00"),
]


class BargheManClient:
    """Client for querying Iranian Tavanir / SAAPA blackout timetables."""

    def __init__(
        self,
        mode: str = "mock",
        base_url: str = "https://uiapi.saapa.ir",
        username: str | None = None,
        password: str | None = None,
        timeout: float = 5.0,
        session: requests.Session | None = None,
    ):
        if mode not in ("mock", "live"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'mock' or 'live'.")

        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = float(timeout)
        self.session = session or requests.Session()

        # Token caching state
        self._token: str | None = None
        self._token_expires_at: float = 0.0

        # Mock schedule store: bill_id -> list of outage dicts
        self._mock_schedules: dict[str, list[dict[str, Any]]] = {}

    # -------------------------------------------------------------------------
    # Authentication & Token Management
    # -------------------------------------------------------------------------
    def authenticate(self, force_refresh: bool = False) -> str:
        """Acquires or refreshes the Bearer token."""
        now = time.time()
        if not force_refresh and self._token and now < (self._token_expires_at - 60):
            return self._token

        if self.mode == "mock":
            self._token = "mock-jwt-saapa-bearer-token-iran-tavanir"
            self._token_expires_at = now + 3600.0
            return self._token

        # Live mode authentication
        login_url = f"{self.base_url}/api/v1/auth/token"
        payload = {
            "username": self.username or "telecom_operator",
            "password": self.password or "",
            "client_type": "ORAN_SMO_CONTROLLER",
        }
        try:
            resp = self.session.post(login_url, json=payload, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise BargheManNetworkError(f"Failed to connect to SAAPA auth endpoint: {e}") from e

        if resp.status_code == 401:
            raise BargheManAuthError("Invalid credentials for SAAPA API.")
        if resp.status_code != 200:
            raise BargheManAPIError(f"SAAPA auth failed with HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        token = data.get("access_token") or data.get("token") or data.get("data", {}).get("token")
        if not token:
            raise BargheManAPIError(f"No token returned in SAAPA response: {data}")

        expires_in = float(data.get("expires_in", 3600))
        self._token = token
        self._token_expires_at = now + expires_in
        return self._token

    # -------------------------------------------------------------------------
    # Mock Schedule Ingestion
    # -------------------------------------------------------------------------
    def load_mock_from_dataframe(self, df: pd.DataFrame):
        """Populates the mock timetable database directly from a telecom telemetry dataframe."""
        self._mock_schedules.clear()
        dt_s = 300

        for site_id in df["site_id"].unique():
            site_df = df[df["site_id"] == site_id].sort_values("timestamp").reset_index(drop=True)
            feeder_id = site_df["feeder_id"].iloc[0] if "feeder_id" in site_df else "F_1"

            sched_blocks = []
            in_block = False
            start_idx = 0
            for idx, r in site_df.iterrows():
                if r["blackout_schedule_flag"] == 1:
                    if not in_block:
                        in_block = True
                        start_idx = idx
                else:
                    if in_block:
                        in_block = False
                        sched_blocks.append((start_idx, idx - 1))
            if in_block:
                sched_blocks.append((start_idx, len(site_df) - 1))

            outages = []
            for s_idx, e_idx in sched_blocks:
                start_ts = site_df.loc[s_idx, "timestamp"]
                end_ts = site_df.loc[e_idx, "timestamp"] + pd.Timedelta(seconds=dt_s)
                dur_min = int((end_ts - start_ts).total_seconds() / 60)

                outage_record = {
                    "bill_id": site_id,
                    "outage_id": f"SAAPA-{start_ts.strftime('%Y%m%d-%H%M')}-{feeder_id}",
                    "feeder_id": feeder_id,
                    "district_code": "THR-DISTRICT",
                    "start_time": start_ts.isoformat(),
                    "end_time": end_ts.isoformat(),
                    "duration_minutes": dur_min,
                    "reason_code": "LOAD_MANAGEMENT_TAVANIR",
                    "reason_fa": "مدیریت اضطراری بار شبکه توزیع",
                    "status": "CONFIRMED",
                }
                outages.append(outage_record)

            self._mock_schedules[site_id] = outages

    # -------------------------------------------------------------------------
    # Core SAAPA Outage Fetching
    # -------------------------------------------------------------------------
    def fetch_planned_outages(
        self,
        bill_id: str,
        from_date: str | datetime.date | pd.Timestamp | None = None,
        to_date: str | datetime.date | pd.Timestamp | None = None,
    ) -> list[dict[str, Any]]:
        """Queries planned outage events conforming to the SAAPA schema.

        bill_id: Iranian 13-digit electricity bill identifier or site ID (e.g. 'SITE_1').
        """
        bill_id = str(bill_id).strip()

        if self.mode == "mock":
            if bill_id not in self._mock_schedules:
                # Generate standard 2-hour Tavanir events and store in cache
                self._mock_schedules[bill_id] = self._generate_synthetic_mock_schedule(bill_id, from_date, to_date)
            records = self._mock_schedules.get(bill_id, [])
            return self._filter_by_dates(records, from_date, to_date)

        # Live Mode
        token = self.authenticate()
        url = f"{self.base_url}/api/v1/outages/planned"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "TelecomDigitalTwin-Arbiter/3.3",
        }
        params: dict[str, str] = {"bill_id": bill_id}
        if from_date:
            params["from_date"] = str(from_date)
        if to_date:
            params["to_date"] = str(to_date)

        try:
            resp = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            if resp.status_code == 401:
                # Retry once after refreshing token
                token = self.authenticate(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                resp = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise BargheManNetworkError(f"Network error querying SAAPA planned outages: {e}") from e

        if resp.status_code != 200:
            raise BargheManAPIError(f"SAAPA query returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data") or data.get("outages") or []
        return []

    # -------------------------------------------------------------------------
    # Operational Helpers for Controller Integration
    # -------------------------------------------------------------------------
    def get_upcoming_outage(
        self,
        bill_id: str,
        current_ts: datetime.datetime | pd.Timestamp,
    ) -> dict[str, Any] | None:
        """Returns the active or upcoming scheduled blackout window for bill_id.

        If currently during an outage, returns the record with remaining_hours.
        If before an outage, returns the record with hours_until_start.
        """
        outages = self.fetch_planned_outages(bill_id)
        if not outages:
            return None

        # Sort outages chronologically by start_time
        sorted_outages = sorted(outages, key=lambda x: pd.to_datetime(x["start_time"]))
        current_ts = pd.to_datetime(current_ts)

        for out in sorted_outages:
            s_ts = pd.to_datetime(out["start_time"])
            e_ts = pd.to_datetime(out["end_time"])

            c_ts = current_ts
            if s_ts.tz is not None and c_ts.tz is None:
                c_ts = c_ts.tz_localize(s_ts.tz)
            elif s_ts.tz is None and c_ts.tz is not None:
                c_ts = c_ts.tz_localize(None)

            # Active outage: strictly while current_ts < e_ts (at e_ts, grid power returns)
            if s_ts <= c_ts < e_ts:
                rem_h = max(0.0, (e_ts - c_ts).total_seconds() / 3600.0)
                res = dict(out)
                res["is_active"] = True
                res["remaining_hours"] = rem_h
                return res

            # Future upcoming outage
            if c_ts < s_ts:
                h_until = (s_ts - c_ts).total_seconds() / 3600.0
                res = dict(out)
                res["is_active"] = False
                res["hours_until_start"] = h_until
                res["duration_hours"] = (e_ts - s_ts).total_seconds() / 3600.0
                return res

            # If c_ts >= e_ts, outage has concluded; check next outage in loop

        return None

    def is_outage_scheduled(
        self,
        bill_id: str,
        current_ts: datetime.datetime | pd.Timestamp,
    ) -> tuple[bool, pd.Timestamp | None]:
        """Returns (True, end_timestamp) if current_ts is within a published blackout, else (False, None)."""
        up = self.get_upcoming_outage(bill_id, current_ts)
        if up and up.get("is_active"):
            return True, pd.to_datetime(up["end_time"])
        return False, None

    def build_schedule_map(
        self,
        df: pd.DataFrame,
        site_id: str = "SITE_1",
        dt_s: int = 300,
    ) -> dict[Any, pd.Timestamp]:
        """Constructs the sched_end_map required by the Receding-Horizon Reserve planner."""
        # Ensure mock DB has latest records if in mock mode
        if self.mode == "mock":
            self.load_mock_from_dataframe(df)

        site_df = df[df["site_id"] == site_id].sort_values("timestamp").reset_index(drop=True)
        outages = self.fetch_planned_outages(site_id)

        sched_end_map: dict[Any, pd.Timestamp] = {}
        df_tz = site_df["timestamp"].dt.tz

        for out in outages:
            s_ts = pd.to_datetime(out["start_time"])
            e_ts = pd.to_datetime(out["end_time"])

            # Harmonize timezones between telemetry dataframe and schedule timestamps
            if df_tz is not None and s_ts.tz is None:
                s_ts = s_ts.tz_localize(df_tz)
                e_ts = e_ts.tz_localize(df_tz)
            elif df_tz is None and s_ts.tz is not None:
                s_ts = s_ts.tz_localize(None)
                e_ts = e_ts.tz_localize(None)
            elif df_tz is not None and s_ts.tz is not None and s_ts.tz != df_tz:
                s_ts = s_ts.tz_convert(df_tz)
                e_ts = e_ts.tz_convert(df_tz)

            # Map both timestamp objects and dataframe indices for O(1) controller lookup
            matching_rows = site_df[(site_df["timestamp"] >= s_ts) & (site_df["timestamp"] < e_ts)]
            for idx, r in matching_rows.iterrows():
                sched_end_map[idx] = e_ts
                sched_end_map[r["timestamp"]] = e_ts

        return sched_end_map

    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------
    def _generate_synthetic_mock_schedule(
        self,
        bill_id: str,
        from_date: Any = None,
        to_date: Any = None,
    ) -> list[dict[str, Any]]:
        """Generates realistic 2-hour Tavanir timetable records for mock testing."""
        start_date = pd.to_datetime(from_date or "2026-01-01").tz_localize(None)
        end_date = pd.to_datetime(to_date or "2026-03-31").tz_localize(None)

        records = []
        cur = start_date
        day_idx = 0
        while cur <= end_date:
            # Outage every 2 days
            if day_idx % 2 == 0:
                slot_s, slot_e = TAVANIR_BLOCKS[day_idx % len(TAVANIR_BLOCKS)]
                sh, sm = map(int, slot_s.split(":"))
                eh, em = map(int, slot_e.split(":"))

                s_dt = pd.Timestamp(
                    year=cur.year, month=cur.month, day=cur.day, hour=sh, minute=sm, tz="Asia/Tehran"
                )
                e_dt = pd.Timestamp(
                    year=cur.year, month=cur.month, day=cur.day, hour=eh, minute=em, tz="Asia/Tehran"
                )

                records.append({
                    "bill_id": bill_id,
                    "outage_id": f"SAAPA-{s_dt.strftime('%Y%m%d-%H%M')}-F1",
                    "feeder_id": "F_1",
                    "district_code": "THR-02",
                    "start_time": s_dt.isoformat(),
                    "end_time": e_dt.isoformat(),
                    "duration_minutes": 120,
                    "reason_code": "LOAD_MANAGEMENT_TAVANIR",
                    "reason_fa": "مدیریت اضطراری بار شبکه",
                    "status": "CONFIRMED",
                })
            cur += pd.Timedelta(days=1)
            day_idx += 1
        self._mock_schedules[bill_id] = records
        return records

    def _filter_by_dates(
        self,
        records: list[dict[str, Any]],
        from_date: Any = None,
        to_date: Any = None,
    ) -> list[dict[str, Any]]:
        if not from_date and not to_date:
            return records

        filtered = []
        from_ts = pd.to_datetime(from_date) if from_date is not None else None

        to_ts = None
        if to_date is not None:
            # If to_date is given as date or date string (YYYY-MM-DD), extend to end of day
            if isinstance(to_date, datetime.date) and not isinstance(to_date, datetime.datetime):
                to_ts = pd.to_datetime(to_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            elif isinstance(to_date, str) and len(to_date.strip()) <= 10:
                to_ts = pd.to_datetime(to_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            else:
                to_ts = pd.to_datetime(to_date)

        for r in records:
            s_ts = pd.to_datetime(r["start_time"])
            e_ts = pd.to_datetime(r["end_time"])
            f_ts = from_ts
            t_ts = to_ts

            if s_ts.tz is not None:
                if f_ts is not None and f_ts.tz is None:
                    f_ts = f_ts.tz_localize(s_ts.tz)
                if t_ts is not None and t_ts.tz is None:
                    t_ts = t_ts.tz_localize(s_ts.tz)
            else:
                if f_ts is not None and f_ts.tz is not None:
                    f_ts = f_ts.tz_localize(None)
                if t_ts is not None and t_ts.tz is not None:
                    t_ts = t_ts.tz_localize(None)

            # Overlap check: outage interval [s_ts, e_ts] intersects query window [f_ts, t_ts]
            if f_ts is not None and e_ts <= f_ts:
                continue
            if t_ts is not None and s_ts > t_ts:
                continue
            filtered.append(r)
        return filtered
