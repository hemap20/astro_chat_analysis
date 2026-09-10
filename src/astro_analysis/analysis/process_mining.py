"""Stage 4.8: process mining.

Exports the tagged session data as an event log (case ID = session_id,
activity = per-message state label reused from the Markov module, timestamp
= message order) and discovers a Directly-Follows Graph per source group,
filtered to the top-N most frequent activities/paths to keep the map
readable. Rendered directly via pm4py (requires the system `dot` binary from
Graphviz).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pm4py

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.markov import label_states
from astro_analysis.analysis.plotting import CATEGORICAL, SEQUENTIAL_BLUE, new_fig, save

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


def plot_process_map_networkx(dfg: dict, start_acts: dict, end_acts: dict, title: str, path, seed=config.RANDOM_SEED):
    """Matplotlib/networkx fallback process map -- no system Graphviz needed.
    Node size ~ activity frequency (from edge weights touching it), edge
    width/opacity ~ transition frequency."""
    G = nx.DiGraph()
    for (src, dst), freq in dfg.items():
        G.add_edge(src, dst, weight=freq)
    if G.number_of_nodes() == 0:
        return

    node_freq = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) +
                    sum(d["weight"] for _, _, d in G.in_edges(n, data=True))
                 for n in G.nodes()}
    max_freq = max(node_freq.values()) if node_freq else 1
    max_edge = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)

    pos = nx.spring_layout(G, seed=seed, k=1.4 / max(len(G.nodes()) ** 0.5, 1))

    fig, ax = new_fig(figsize=(11, 9))
    ax.axis("off")

    for src, dst, d in G.edges(data=True):
        w = d["weight"] / max_edge
        ax.annotate("", xy=pos[dst], xytext=pos[src],
                    arrowprops=dict(arrowstyle="-|>", color=CATEGORICAL[0],
                                     alpha=0.25 + 0.6 * w, linewidth=0.5 + 3 * w,
                                     shrinkA=14, shrinkB=14, connectionstyle="arc3,rad=0.08"))

    sizes = [250 + 1400 * (node_freq[n] / max_freq) for n in G.nodes()]
    node_colors = []
    for n in G.nodes():
        if n in start_acts:
            node_colors.append(CATEGORICAL[2])
        elif n in end_acts:
            node_colors.append(CATEGORICAL[7])
        else:
            node_colors.append(CATEGORICAL[0])
    ax.scatter([pos[n][0] for n in G.nodes()], [pos[n][1] for n in G.nodes()],
               s=sizes, c=node_colors, alpha=0.9, zorder=3, edgecolors="white", linewidths=1.2)
    for n in G.nodes():
        x, y = pos[n]
        ax.annotate(n, (x, y), xytext=(0, 11), textcoords="offset points",
                    fontsize=7.5, ha="center", va="bottom", zorder=4, color="#0b0b0b",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.75))

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=CATEGORICAL[2], label="session-start activity", markersize=9),
        plt.Line2D([0], [0], marker="o", linestyle="", color=CATEGORICAL[7], label="session-end activity", markersize=9),
        plt.Line2D([0], [0], marker="o", linestyle="", color=CATEGORICAL[0], label="other", markersize=9),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8)
    ax.set_title(title)
    save(fig, path)


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
            print(f"[process_mining] pm4py/Graphviz render unavailable for {name} "
                  f"({e}); falling back to matplotlib/networkx renderer")
            plot_process_map_networkx(dfg, start_acts, end_acts,
                                       f"Process map (top-{TOP_N_ACTIVITIES} states): {name}",
                                       OUT_DIR / f"process_map_{name}.png")

    print(f"Stage 4.8 process mining written to {OUT_DIR}")


if __name__ == "__main__":
    run()
