"""Stage 1: parse raw conversation CSVs into normalized messages + sessions tables.

Each input CSV = all sessions between one user and one astrologer. Rows are
typically reverse-chronological; we sort ascending and verify per-file. A
system sender (ID "0") marks session boundaries via "Welcome to AstroLokal"
(start) and "Chat has ended" (end) messages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from astro_analysis import config

PROFILE_FIELD_PATTERN = re.compile(
    r"^\s*(Name|Gender|DOB|TOB|POB)\s*-\s*(.*)$", re.IGNORECASE | re.MULTILINE
)


@dataclass
class ParseIssue:
    source_folder: str
    file: str
    issue: str
    detail: str = ""


ISSUES: list[ParseIssue] = []


def _is_system(sender_id: str) -> bool:
    return str(sender_id).strip() == config.SYSTEM_SENDER_ID


def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
    low = (text or "").strip().lower()
    return any(m in low for m in markers)


def parse_user_profile(message: str) -> dict:
    """Extract Name/Gender/DOB/TOB/POB fields from a structured opening message."""
    profile = {}
    for m in PROFILE_FIELD_PATTERN.finditer(message or ""):
        key, val = m.group(1).title(), m.group(2).strip()
        if val and val != "-":
            profile[key] = val
    return profile


def _read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    expected = {"Date", "Time", "Sender Name", "Sender ID", "Message"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing} in {path}")
    return df


def _sort_and_verify(df: pd.DataFrame, source_folder: str, fname: str) -> pd.DataFrame:
    df["_timestamp"] = pd.to_datetime(
        df["Date"].str.strip() + " " + df["Time"].str.strip(), errors="coerce"
    )
    n_bad = df["_timestamp"].isna().sum()
    if n_bad:
        ISSUES.append(
            ParseIssue(source_folder, fname, "unparseable_timestamp", f"{n_bad} rows")
        )
        df = df.dropna(subset=["_timestamp"])

    # Rows are expected to be reverse-chronological, but near-simultaneous
    # messages (two people typing in the same second) can be interleaved
    # out of strict order upstream. Sort ascending regardless; only flag
    # when a large fraction of consecutive rows disagree with either
    # direction, which would suggest the file isn't simply reverse-sorted.
    diffs = df["_timestamp"].diff().dt.total_seconds().dropna()
    frac_ascending_steps = (diffs > 0).mean() if len(diffs) else 0.0
    frac_descending_steps = (diffs < 0).mean() if len(diffs) else 0.0
    df_sorted = df.sort_values("_timestamp", kind="stable").reset_index(drop=True)

    is_reverse_chrono = frac_descending_steps >= 0.9
    is_forward_chrono = frac_ascending_steps >= 0.9
    if not is_reverse_chrono and not is_forward_chrono:
        ISSUES.append(
            ParseIssue(
                source_folder,
                fname,
                "unexpected_raw_order",
                f"only {frac_descending_steps:.0%} descending / {frac_ascending_steps:.0%} ascending steps",
            )
        )
    return df_sorted


def _split_sessions(df: pd.DataFrame, source_folder: str, fname: str) -> list[pd.DataFrame]:
    """Split a chronologically-sorted message dataframe into per-session chunks
    using system start/end markers. Falls back to treating the whole file as one
    session (flagged) if no markers are found at all."""
    start_idxs, end_idxs = [], []
    for i, row in df.iterrows():
        if not _is_system(row["Sender ID"]):
            continue
        msg = row["Message"]
        if _matches_any(msg, config.SESSION_START_MARKERS):
            start_idxs.append(i)
        elif _matches_any(msg, config.SESSION_END_MARKERS):
            end_idxs.append(i)

    if not start_idxs and not end_idxs:
        ISSUES.append(
            ParseIssue(
                source_folder, fname, "no_session_markers_found",
                "treating entire file as a single session",
            )
        )
        return [df]

    if not end_idxs:
        ISSUES.append(ParseIssue(source_folder, fname, "no_end_markers", ""))

    boundaries = sorted(set(start_idxs))
    # The first session doesn't always open with an explicit "Welcome to
    # AstroLokal" marker (e.g. the file's coverage window starts mid-session,
    # or the export just omits it). Whenever the earliest known start marker
    # isn't already at row 0, synthesize a boundary at the file start so
    # leading messages are kept as session 1 rather than silently dropped.
    if not boundaries or boundaries[0] != df.index[0]:
        ISSUES.append(
            ParseIssue(
                source_folder, fname, "no_start_marker_for_first_session",
                "first session has no explicit welcome marker; treating file start as its boundary",
            )
        )
        boundaries = [df.index[0]] + boundaries
    sessions = []
    for k, start in enumerate(boundaries):
        stop = boundaries[k + 1] if k + 1 < len(boundaries) else df.index[-1] + 1
        chunk = df.loc[start:stop - 1]
        if len(chunk) > 1:  # more than just the marker row
            sessions.append(chunk)

    if not sessions:
        ISSUES.append(
            ParseIssue(source_folder, fname, "empty_sessions_after_split", "")
        )
        return [df]

    # A session chunk spanning an unusually large internal gap, with no marker
    # in between, likely hides a real session boundary we have no marker to
    # detect. Flag it rather than silently guessing where to cut.
    for chunk in sessions:
        content = chunk[~chunk["Sender ID"].apply(_is_system)]
        if len(content) < 2:
            continue
        gaps = content["_timestamp"].diff().dt.total_seconds().dropna() / 3600.0
        max_gap = gaps.max() if len(gaps) else 0.0
        if max_gap >= config.UNMARKED_SESSION_GAP_HOURS:
            ISSUES.append(
                ParseIssue(
                    source_folder, fname, "large_internal_gap_possible_unmarked_boundary",
                    f"{max_gap:.1f}h gap inside one detected session with no marker",
                )
            )

    # flag if the last session in the file never saw an explicit "Chat has ended"
    last_chunk = sessions[-1]
    if not any(
        _is_system(r["Sender ID"]) and _matches_any(r["Message"], config.SESSION_END_MARKERS)
        for _, r in last_chunk.iterrows()
    ):
        ISSUES.append(
            ParseIssue(source_folder, fname, "session_missing_end_marker", "last session in file")
        )

    return sessions


def _identify_roles(df: pd.DataFrame, source_folder: str, fname: str) -> tuple[str, str]:
    """Return (user_id, astrologer_id) using Sender ID, not name."""
    non_system_ids = df.loc[~df["Sender ID"].apply(_is_system), "Sender ID"].unique().tolist()
    m = re.match(r"conversation_(\d+)_(\d+)\.csv$", fname)
    filename_user_id = m.group(1) if m else None

    if filename_user_id and filename_user_id in non_system_ids:
        user_id = filename_user_id
    elif len(non_system_ids) >= 1:
        # fall back: pick the id with the most messages as the user (heuristic)
        counts = df.loc[~df["Sender ID"].apply(_is_system), "Sender ID"].value_counts()
        user_id = counts.index[0]
        ISSUES.append(
            ParseIssue(source_folder, fname, "user_id_inferred_not_from_filename", str(user_id))
        )
    else:
        raise ValueError(f"no non-system senders found in {fname}")

    astro_ids = [sid for sid in non_system_ids if sid != user_id]
    if len(astro_ids) == 0:
        ISSUES.append(ParseIssue(source_folder, fname, "no_astrologer_sender_found", ""))
        astrologer_id = None
    elif len(astro_ids) > 1:
        counts = df.loc[df["Sender ID"].isin(astro_ids), "Sender ID"].value_counts()
        astrologer_id = counts.index[0]
        ISSUES.append(
            ParseIssue(
                source_folder, fname, "multiple_astrologer_ids_found",
                f"chose {astrologer_id} from {astro_ids}",
            )
        )
    else:
        astrologer_id = astro_ids[0]

    return user_id, astrologer_id


def parse_file(path: Path, source_folder: str) -> tuple[list[dict], list[dict]]:
    """Parse a single CSV. Returns (message_rows, session_rows)."""
    fname = path.name
    raw = _read_raw_csv(path)
    if raw.empty:
        ISSUES.append(ParseIssue(source_folder, fname, "empty_file", ""))
        return [], []

    df = _sort_and_verify(raw, source_folder, fname)
    user_id, astrologer_id = _identify_roles(df, source_folder, fname)

    session_chunks = _split_sessions(df, source_folder, fname)

    message_rows, session_rows = [], []
    prev_session_end: Optional[pd.Timestamp] = None
    user_profile: dict = {}

    for session_number, chunk in enumerate(session_chunks, start=1):
        content_rows = chunk[~chunk["Sender ID"].apply(_is_system)]
        if content_rows.empty:
            continue

        session_start_time = content_rows["_timestamp"].min()
        session_end_time = content_rows["_timestamp"].max()

        # opening message of session 1 usually carries structured profile details
        if session_number == 1:
            user_rows_sorted = content_rows[content_rows["Sender ID"] == user_id]
            if not user_rows_sorted.empty:
                opening_msg = user_rows_sorted.iloc[0]["Message"]
                user_profile = parse_user_profile(opening_msg)

        gap_since_previous = (
            None if prev_session_end is None
            else (session_start_time - prev_session_end).total_seconds() / 3600.0
        )

        astro_msgs = content_rows[content_rows["Sender ID"] == astrologer_id]
        user_msgs = content_rows[content_rows["Sender ID"] == user_id]

        # session-level ending_type is left for Stage 2 tagging; record raw signal
        end_marker_present = any(
            _is_system(r["Sender ID"]) and _matches_any(r["Message"], config.SESSION_END_MARKERS)
            for _, r in chunk.iterrows()
        )

        session_id = f"{source_folder}::{fname}::s{session_number}"

        for order, (_, row) in enumerate(content_rows.iterrows()):
            is_astrologer = row["Sender ID"] == astrologer_id
            message_rows.append({
                "source_folder": source_folder,
                "file": fname,
                "session_id": session_id,
                "user_id": user_id,
                "astrologer_id": astrologer_id,
                "session_number": session_number,
                "message_order": order,
                "timestamp": row["_timestamp"],
                "sender_id": row["Sender ID"],
                "sender_name": row["Sender Name"],
                "role": "astrologer" if is_astrologer else "user",
                "message": row["Message"],
            })

        session_rows.append({
            "source_folder": source_folder,
            "file": fname,
            "session_id": session_id,
            "user_id": user_id,
            "astrologer_id": astrologer_id,
            "session_number": session_number,
            "session_start_time": session_start_time,
            "session_end_time": session_end_time,
            "session_duration_minutes": (session_end_time - session_start_time).total_seconds() / 60.0,
            "gap_since_previous_session_hours": gap_since_previous,
            "message_count_user": len(user_msgs),
            "message_count_astrologer": len(astro_msgs),
            "end_marker_present": end_marker_present,
            **{f"user_{k.lower()}": v for k, v in user_profile.items()},
        })

        prev_session_end = session_end_time

    return message_rows, session_rows


def parse_all(source_folders: Optional[dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_folders = source_folders or config.SOURCE_FOLDERS
    ISSUES.clear()
    all_messages, all_sessions = [], []

    for source_folder, folder_path in source_folders.items():
        if not folder_path.exists():
            ISSUES.append(ParseIssue(source_folder, "", "folder_missing", str(folder_path)))
            continue
        csv_files = sorted(folder_path.glob("*.csv"))
        for path in csv_files:
            try:
                msgs, sessions = parse_file(path, source_folder)
                all_messages.extend(msgs)
                all_sessions.extend(sessions)
            except Exception as e:  # noqa: BLE001
                ISSUES.append(ParseIssue(source_folder, path.name, "parse_exception", str(e)))

    messages_df = pd.DataFrame(all_messages)
    sessions_df = pd.DataFrame(all_sessions)

    # total_sessions_for_user / returned flag, computed per (source_folder, user_id, file)
    if not sessions_df.empty:
        totals = sessions_df.groupby(["source_folder", "file"])["session_number"].transform("max")
        sessions_df["total_sessions_in_file"] = totals
        sessions_df["returned"] = sessions_df["total_sessions_in_file"] >= 2

    return messages_df, sessions_df


def save_issues_log():
    issues_df = pd.DataFrame([i.__dict__ for i in ISSUES])
    issues_df.to_csv(config.PARSE_ISSUES_LOG, index=False)
    return issues_df


def run():
    messages_df, sessions_df = parse_all()
    messages_df.to_parquet(config.MESSAGES_TABLE_PATH, index=False)
    sessions_df.to_parquet(config.SESSIONS_TABLE_PATH, index=False)
    issues_df = save_issues_log()

    print(f"Parsed {len(messages_df)} messages across {sessions_df['session_id'].nunique()} sessions")
    print(f"Sessions table: {len(sessions_df)} rows -> {config.SESSIONS_TABLE_PATH}")
    print(f"Messages table: {len(messages_df)} rows -> {config.MESSAGES_TABLE_PATH}")
    print(f"Logged {len(issues_df)} parse issues -> {config.PARSE_ISSUES_LOG}")
    if not issues_df.empty:
        print(issues_df["issue"].value_counts())
    return messages_df, sessions_df


if __name__ == "__main__":
    run()
