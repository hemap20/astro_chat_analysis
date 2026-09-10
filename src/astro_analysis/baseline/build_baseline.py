"""Stage 3: outcome-grounded baseline construction.

The real baseline split is outcome-grounded: "returned" (2+ sessions total)
vs. "not_returned" (exactly 1 session total). Human_single is the only source
with genuine not_returned users (unfiltered, so it actually contains people
who left after one session). Human_Astro is a hand-picked "best technique"
sample with no not_returned users at all (every file is many-session by
construction) -- so it can never inform the not_returned side, but every one
of its users unambiguously returned, so per product decision it is folded
into baseline_returned to grow that group's sample size. Human_single's own
returned users are included too. baseline_not_returned draws exclusively from
Human_single's genuine single-session users.

Every later analysis stage should pull from the reusable views this module
produces rather than recomputing the split.

"Session count" here is by explicit start/end marker detection (see
astro_analysis.ingestion.parser). Some Human_single files carry a flagged
large unmarked internal time gap that markers alone can't resolve into a
session boundary (config.UNMARKED_SESSION_GAP_HOURS) -- per product decision,
this baseline uses the marker-based counts as-is rather than guessing at
those boundaries; see data/logs/parse_issues.csv for which files are flagged.
"""
from __future__ import annotations

import pandas as pd

from astro_analysis import config

HUMAN_SINGLE_FOLDER = "Human_Single"
HUMAN_ASTRO_FOLDER = "Human_Astro"

BASELINE_RETURNED_SESSIONS_PATH = config.PROCESSED_DIR / "baseline_returned_sessions.parquet"
BASELINE_NOT_RETURNED_SESSIONS_PATH = config.PROCESSED_DIR / "baseline_not_returned_sessions.parquet"
BASELINE_RETURNED_MESSAGES_PATH = config.PROCESSED_DIR / "baseline_returned_messages.parquet"
BASELINE_NOT_RETURNED_MESSAGES_PATH = config.PROCESSED_DIR / "baseline_not_returned_messages.parquet"
BASELINE_SUMMARY_PATH = config.RESULTS_DIR / "baseline_summary.json"


def build(sessions_df: pd.DataFrame, messages_df: pd.DataFrame) -> dict:
    hs_sessions = sessions_df[sessions_df["source_folder"] == HUMAN_SINGLE_FOLDER].copy()
    ha_sessions = sessions_df[sessions_df["source_folder"] == HUMAN_ASTRO_FOLDER].copy()
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

    hs_returned_files = set(first_sessions.loc[first_sessions["outcome_group"] == "returned", "file"])
    hs_not_returned_files = set(first_sessions.loc[first_sessions["outcome_group"] == "not_returned", "file"])
    ha_files = set(ha_sessions["file"].unique())  # every Human_Astro user is, by construction, returned

    def _subset(df, folder, files):
        return df[(df["source_folder"] == folder) & (df["file"].isin(files))].copy()

    baseline_returned_sessions = pd.concat([
        _subset(sessions_df, HUMAN_SINGLE_FOLDER, hs_returned_files),
        _subset(sessions_df, HUMAN_ASTRO_FOLDER, ha_files),
    ], ignore_index=True)
    baseline_not_returned_sessions = _subset(sessions_df, HUMAN_SINGLE_FOLDER, hs_not_returned_files)

    baseline_returned_messages = pd.concat([
        _subset(messages_df, HUMAN_SINGLE_FOLDER, hs_returned_files),
        _subset(messages_df, HUMAN_ASTRO_FOLDER, ha_files),
    ], ignore_index=True)
    baseline_not_returned_messages = _subset(messages_df, HUMAN_SINGLE_FOLDER, hs_not_returned_files)

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
        "n_users_returned": len(hs_returned_files) + len(ha_files),
        "n_users_returned_human_single": len(hs_returned_files),
        "n_users_returned_human_astro": len(ha_files),
        "n_users_not_returned": len(hs_not_returned_files),
        "n_sessions_returned": int(baseline_returned_sessions["session_id"].nunique()),
        "n_sessions_not_returned": int(baseline_not_returned_sessions["session_id"].nunique()),
        "warning": (
            "baseline_not_returned is empty or near-empty; Human_single needs "
            "more genuine single-session users added before this baseline is "
            "trustworthy for comparison."
            if len(hs_not_returned_files) == 0 else None
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
