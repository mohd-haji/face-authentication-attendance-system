"""
Attendance logic: punch-in / punch-out, working hours calculation.
Handles edge cases: missed punch-out, multiple sessions per day.
"""

from datetime import datetime, date, timedelta
from typing import Optional
from dataclasses import dataclass

from utils import get_storage


# ---------------------------------------------------------------------------
# Punch state
# ---------------------------------------------------------------------------

@dataclass
class PunchState:
    """Current punch state for a user on a given date."""
    user_id: str
    name: str
    is_punched_in: bool
    last_punch_in_time: Optional[datetime]
    last_punch_out_time: Optional[datetime]


def get_punch_state(user_id: str, name: str, for_date: Optional[date] = None) -> PunchState:
    """
    Determine if user is currently punched in based on attendance records.
    Rule: odd number of punches today (last is punch-in) => punched in;
          even (last is punch-out) or no punches => punched out.
    """
    for_date = for_date or date.today()
    start = datetime.combine(for_date, datetime.min.time())
    end = start + timedelta(days=1)
    storage = get_storage()
    records = storage.get_records(user_id=user_id, date_from=start, date_to=end)
    punch_ins = [r for r in records if (r.get("punch_type") or "").lower() == "in"]
    punch_outs = [r for r in records if (r.get("punch_type") or "").lower() == "out"]
    last_in = None
    last_out = None
    for r in records:
        ts = r.get("timestamp")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if "+" in str(ts) or "Z" in str(ts):
                dt = dt.replace(tzinfo=None)
        except Exception:
            continue
        if (r.get("punch_type") or "").lower() == "in":
            last_in = dt
        elif (r.get("punch_type") or "").lower() == "out":
            last_out = dt
    # Punched in iff last punch today is IN (and no OUT after it)
    is_in = False
    if last_in is not None and last_out is None:
        is_in = True
    elif last_in is not None and last_out is not None:
        is_in = last_in > last_out
    return PunchState(
        user_id=user_id,
        name=name,
        is_punched_in=is_in,
        last_punch_in_time=last_in,
        last_punch_out_time=last_out,
    )


def get_next_punch_type(user_id: str, name: str, for_date: Optional[date] = None) -> str:
    """Returns 'in' or 'out' for the next punch to record."""
    state = get_punch_state(user_id, name, for_date)
    return "out" if state.is_punched_in else "in"


# ---------------------------------------------------------------------------
# Session duration & daily hours
# ---------------------------------------------------------------------------

@dataclass
class Session:
    punch_in: datetime
    punch_out: Optional[datetime]
    duration_minutes: Optional[float]


def sessions_for_date(user_id: str, for_date: Optional[date] = None) -> list[Session]:
    """
    Build ordered list of sessions (punch_in, punch_out) for the given date.
    Handles multiple sessions per day and missed punch-out (open session).
    """
    for_date = for_date or date.today()
    start = datetime.combine(for_date, datetime.min.time())
    end = start + timedelta(days=1)
    storage = get_storage()
    records = storage.get_records(user_id=user_id, date_from=start, date_to=end)
    in_out = []
    for r in records:
        ts = r.get("timestamp")
        pt = (r.get("punch_type") or "").lower()
        if not ts or pt not in ("in", "out"):
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if "+" in str(ts) or "Z" in str(ts):
                dt = dt.replace(tzinfo=None)
        except Exception:
            continue
        in_out.append((dt, pt))
    in_out.sort(key=lambda x: x[0])
    sessions = []
    i = 0
    while i < len(in_out):
        t, pt = in_out[i]
        if pt == "in":
            punch_out = None
            if i + 1 < len(in_out) and in_out[i + 1][1] == "out":
                punch_out = in_out[i + 1][0]
                i += 2
            else:
                i += 1
            dur = None
            if punch_out is not None:
                dur = (punch_out - t).total_seconds() / 60.0
            sessions.append(Session(punch_in=t, punch_out=punch_out, duration_minutes=dur))
        else:
            i += 1
    return sessions


def total_working_minutes_for_date(user_id: str, for_date: Optional[date] = None) -> float:
    """Sum of all session durations for the date. Missed punch-out sessions count 0 for that session."""
    total = 0.0
    for s in sessions_for_date(user_id, for_date):
        if s.duration_minutes is not None:
            total += s.duration_minutes
    return total


def total_working_hours_for_date(user_id: str, for_date: Optional[date] = None) -> float:
    return total_working_minutes_for_date(user_id, for_date) / 60.0


# ---------------------------------------------------------------------------
# Record punch (with session duration on punch-out)
# ---------------------------------------------------------------------------

def record_punch(
    user_id: str,
    name: str,
    punch_type: str,
    timestamp: Optional[datetime] = None,
) -> bool:
    """
    Record a punch-in or punch-out. Prevents duplicate punch-ins and invalid sequences
    (e.g. punch-out when not punched in is still written but get_punch_state logic
    keeps consistency; we enforce: only allow 'in' when not in, 'out' when in).
    Returns True if recorded.
    """
    punch_type = (punch_type or "").lower()
    if punch_type not in ("in", "out"):
        return False
    ts = timestamp or datetime.now()
    state = get_punch_state(user_id, name, ts.date())
    if punch_type == "in" and state.is_punched_in:
        return False  # Duplicate punch-in
    if punch_type == "out" and not state.is_punched_in:
        return False  # Punch-out without punch-in
    session_minutes = None
    if punch_type == "out" and state.last_punch_in_time is not None:
        session_minutes = (ts - state.last_punch_in_time).total_seconds() / 60.0
    storage = get_storage()
    storage.init_schema()
    storage.append_record(
        user_id=user_id,
        name=name,
        punch_type=punch_type,
        timestamp=ts,
        session_minutes=session_minutes,
    )
    return True


def get_attendance_summary(
    user_id: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[dict]:
    """
    Get attendance records as list of dicts with keys:
    user_id, name, punch_type, timestamp, session_minutes.
    """
    storage = get_storage()
    storage.init_schema()
    start = datetime.combine(date_from, datetime.min.time()) if date_from else None
    end = (datetime.combine(date_to, datetime.min.time()) + timedelta(days=1)) if date_to else None
    return storage.get_records(user_id=user_id, date_from=start, date_to=end)
