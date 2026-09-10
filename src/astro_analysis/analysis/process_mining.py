"""Stage 4.8: process mining.

Exports the tagged session data as an event log (case ID = session_id,
activity = per-message state label reused from the Markov module, timestamp
= message order) and discovers a Directly-Follows Graph per source group,
filtered to the top-N most frequent activities/paths to keep the map
readable. Rendered directly via pm4py (requires the system `dot` binary from
Graphviz).
"""
from __future__ import annotations

import pandas as pd
import pm4py

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.markov import label_states

OUT_DIR = config.RESULTS_DIR / "process_mining"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_ACTIVITIES = 10


def build_event_log(messages: pd.DataFrame) -> pd.DataFrame:
    labeled = label_states(messages).sort_values(["session_id", "message_order"])
    log = labeled[["session_id", "state", "timestamp"]].rename(columns={
        "session_id": "case:concept:name",
        "state": "concept:name",
        "timestamp": "time:timestamp",
    })
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    # pm4py needs strictly increasing timestamps per case for ordering; message
    # order within a session is already correct, so add a tiny per-row offset
    # to break any same-second ties instead of letting them re-sort.
    log["time:timestamp"] = log["time:timestamp"] + pd.to_timedelta(
        log.groupby("case:concept:name").cumcount(), unit="ms")
    return log


def filter_top_activities(log: pd.DataFrame, top_n=TOP_N_ACTIVITIES) -> pd.DataFrame:
    top_activities = log["concept:name"].value_counts().head(top_n).index
    return log[log["concept:name"].isin(top_activities)]


def run():
    group_frames = groups_mod.get_groups()

    for name, frames in group_frames.items():
        log = build_event_log(frames["messages"])
        if log.empty:
            continue
        log.to_csv(OUT_DIR / f"event_log_{name}.csv", index=False)

        filtered = filter_top_activities(log)
        dfg, start_acts, end_acts = pm4py.discover_dfg(
            filtered, case_id_key="case:concept:name",
            activity_key="concept:name", timestamp_key="time:timestamp")

        pd.Series(dfg).rename("frequency").rename_axis(["from", "to"]).reset_index().to_csv(
            OUT_DIR / f"dfg_edges_{name}.csv", index=False)
        pd.Series(start_acts).rename("frequency").to_csv(OUT_DIR / f"start_activities_{name}.csv")
        pd.Series(end_acts).rename("frequency").to_csv(OUT_DIR / f"end_activities_{name}.csv")

        try:
            pm4py.save_vis_dfg(dfg, start_acts, end_acts,
                                file_path=str(OUT_DIR / f"process_map_{name}.png"))
        except Exception as e:  # noqa: BLE001
            print(f"[process_mining] could not render process map for {name} "
                  f"(is Graphviz's `dot` installed?): {e}")

    print(f"Stage 4.8 process mining written to {OUT_DIR}")


if __name__ == "__main__":
    run()
