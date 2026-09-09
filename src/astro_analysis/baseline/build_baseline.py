"""Stage 3: outcome-grounded baseline construction from Human_Single.

Human_Astro is a hand-picked "best technique" reference and must NOT be used
as the retention baseline. The real baseline comes from Human_Single: split
each user's FIRST session into "returned" (2+ sessions total) vs.
"not_returned" (exactly 1 session total). Every later analysis stage should
pull from the reusable views this module produces rather than recomputing
the split.

NOTE (as of the initial build): the Human_Single sample available at build
time contained only 9 files, all with 5-10 sessions -- i.e. zero
single-session ("not_returned") users. This module still produces both views
and will correctly populate `baseline_not_returned` once a fuller
Human_Single export (including 1-2 session users) is dropped in; until then
that view may be empty and downstream comparisons involving it should be
treated as provisional / flagged.
"""
from __future__ import annotations

import pandas as pd

from astro_analysis import config

HUMAN_SINGLE_FOLDER = "Human_Single"

BASELINE_RETURNED_SESSIONS_PATH = config.PROCESSED_DIR / "baseline_returned_sessions.parquet"
BASELINE_NOT_RETURNED_SESSIONS_PATH = config.PROCESSED_DIR / "baseline_not_returned_sessions.parquet"
BASELINE_RETURNED_MESSAGES_PATH = config.PROCESSED_DIR / "baseline_returned_messages.parquet"
BASELINE_NOT_RETURNED_MESSAGES_PATH = config.PROCESSED_DIR / "baseline_not_returned_messages.parquet"
BASELINE_SUMMARY_PATH = config.RESULTS_DIR / "baseline_summary.json"


def build(sessions_df: pd.DataFrame, messages_df: pd.DataFrame) -> dict:
    hs_sessions = sessions_df[sessions_df["source_folder"] == HUMAN_SINGLE_FOLDER].copy()
    if hs_sessions.empty:
        raise ValueError(
            f"No sessions found for source_folder={HUMAN_SINGLE_FOLDER!r}; "
            "check config.SOURCE_FOLDERS and that Stage 1/2 ran on it."
        )

    # one row per user (per file) describing outcome, keyed off session 1
    first_sessions = hs_sessions[hs_sessions["session_number"] == 1].copy()
    first_sessions["outcome_group"] = first_sessions["returned"].map(
        {True: "returned", False: "not_returned"}
    )

    returned_files = set(first_sessions.loc[first_sessions["outcome_group"] == "returned", "file"])
    not_returned_files = set(first_sessions.loc[first_sessions["outcome_group"] == "not_returned", "file"])

    def _subset(df, files, key="file"):
        return df[df["file"].isin(files) & (df["source_folder"] == HUMAN_SINGLE_FOLDER)].copy()

    baseline_returned_sessions = _subset(hs_sessions, returned_files)
    baseline_not_returned_sessions = _subset(hs_sessions, not_returned_files)
    baseline_returned_messages = _subset(messages_df, returned_files)
    baseline_not_returned_messages = _subset(messages_df, not_returned_files)

    for df, tag in (
        (baseline_returned_sessions, "returned"),
        (baseline_not_returned_sessions, "not_returned"),
        (baseline_returned_messages, "returned"),
        (baseline_not_returned_messages, "not_returned"),
    ):
        df["baseline_group"] = tag

    baseline_returned_sessions.to_parquet(BASELINE_RETURNED_SESSIONS_PATH, index=False)
    baseline_not_returned_sessions.to_parquet(BASELINE_NOT_RETURNED_SESSIONS_PATH, index=False)
    baseline_returned_messages.to_parquet(BASELINE_RETURNED_MESSAGES_PATH, index=False)
    baseline_not_returned_messages.to_parquet(BASELINE_NOT_RETURNED_MESSAGES_PATH, index=False)

    summary = {
        "n_users_total": int(first_sessions["file"].nunique()),
        "n_users_returned": len(returned_files),
        "n_users_not_returned": len(not_returned_files),
        "n_sessions_returned": int(baseline_returned_sessions["session_id"].nunique()),
        "n_sessions_not_returned": int(baseline_not_returned_sessions["session_id"].nunique()),
        "warning": (
            "baseline_not_returned is empty or near-empty; the Human_Single "
            "sample used to build this needs single-session users added "
            "before this baseline is trustworthy for comparison."
            if len(not_returned_files) == 0 else None
        ),
    }
    pd.Series(summary).to_json(BASELINE_SUMMARY_PATH, indent=2)
    return summary


def load_baseline() -> dict:
    """Reusable accessor for downstream analysis modules."""
    return {
        "returned_sessions": pd.read_parquet(BASELINE_RETURNED_SESSIONS_PATH),
        "not_returned_sessions": pd.read_parquet(BASELINE_NOT_RETURNED_SESSIONS_PATH),
        "returned_messages": pd.read_parquet(BASELINE_RETURNED_MESSAGES_PATH),
        "not_returned_messages": pd.read_parquet(BASELINE_NOT_RETURNED_MESSAGES_PATH),
    }


def run():
    sessions_df = pd.read_parquet(config.SESSIONS_TAGGED_PATH)
    messages_df = pd.read_parquet(config.MESSAGES_TAGGED_PATH)
    summary = build(sessions_df, messages_df)
    print("Baseline summary:", summary)
    return summary


if __name__ == "__main__":
    run()
