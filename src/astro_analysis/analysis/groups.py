"""Shared comparison-group loader for every Stage 4 analysis module.

Four groups are compared throughout: baseline_returned, baseline_not_returned
(from Stage 3), and Sitara-1 / Sitara-2 (straight from the tagged tables,
filtered by source_folder). Every module should get its data from here rather
than re-deriving the split, so the groups stay consistent across analyses.
"""
from __future__ import annotations

import pandas as pd

from astro_analysis import config
from astro_analysis.baseline import build_baseline

GROUP_NAMES = ["baseline_returned", "baseline_not_returned", "Sitara-1", "Sitara-2"]


def session_number_bucket(session_number) -> str:
    n = int(session_number)
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def load_all_tagged() -> tuple[pd.DataFrame, pd.DataFrame]:
    messages_df = pd.read_parquet(config.MESSAGES_TAGGED_PATH)
    sessions_df = pd.read_parquet(config.SESSIONS_TAGGED_PATH)
    sessions_df["session_bucket"] = sessions_df["session_number"].map(session_number_bucket)
    messages_df["session_bucket"] = messages_df["session_number"].map(session_number_bucket)
    return messages_df, sessions_df


def get_groups() -> dict:
    """Returns {group_name: {"sessions": df, "messages": df}} for the four
    standard comparison groups."""
    messages_df, sessions_df = load_all_tagged()
    baseline = build_baseline.load_baseline()

    groups = {
        "baseline_returned": {
            "sessions": baseline["returned_sessions"].assign(
                session_bucket=lambda d: d["session_number"].map(session_number_bucket)),
            "messages": baseline["returned_messages"].assign(
                session_bucket=lambda d: d["session_number"].map(session_number_bucket)),
        },
        "baseline_not_returned": {
            "sessions": baseline["not_returned_sessions"].assign(
                session_bucket=lambda d: d["session_number"].map(session_number_bucket)),
            "messages": baseline["not_returned_messages"].assign(
                session_bucket=lambda d: d["session_number"].map(session_number_bucket)),
        },
    }
    for name in ("Sitara-1", "Sitara-2"):
        groups[name] = {
            "sessions": sessions_df[sessions_df["source_folder"] == name].copy(),
            "messages": messages_df[messages_df["source_folder"] == name].copy(),
        }
    return groups
