"""Stage 4.3: Markov transition modeling.

Builds a transition matrix over a coarse per-message "state" label
(situation_tag -> astrologer_tag -> next situation_tag, collapsed to one state
per message) per source group, then diffs each Sitara version's matrix
against baseline_returned's.

Per-message state is a single label chosen by priority so the matrix stays
small and readable:
  user messages   -> the situation tag that fired (or "user_neutral")
  astrologer msgs -> the forward-pull/relational tag that fired (or
                      "astro_other"), independent of the user-side tags
State priority order (first match wins) keeps this deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.plotting import DIVERGING, SEQUENTIAL_BLUE, new_fig, save

OUT_DIR = config.RESULTS_DIR / "markov"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USER_STATE_PRIORITY = [
    ("pushback_disagreement", "user_pushback"),
    ("disengagement_signal", "user_disengaging"),
    ("emotional_disclosure", "user_emotional_disclosure"),
    ("enthusiastic_engagement", "user_enthusiastic"),
    ("neutral_factual", "user_neutral_factual"),
]
ASTRO_STATE_PRIORITY = [
    ("concrete_dated_prediction", "astro_dated_prediction"),
    ("callback_prior_session", "astro_callback_prior"),
    ("callback_within_session", "astro_callback_within"),
    ("reframe", "astro_reframe"),
    ("validation", "astro_validation"),
    ("self_correction", "astro_self_correction"),
    ("specific_open_thread_named", "astro_open_thread"),
]


def _state_for_row(row) -> str:
    priority = ASTRO_STATE_PRIORITY if row["role"] == "astrologer" else USER_STATE_PRIORITY
    default = "astro_other" if row["role"] == "astrologer" else "user_neutral"
    for col, label in priority:
        val = row.get(col)
        if val is True:
            return label
    return default


def label_states(messages: pd.DataFrame) -> pd.DataFrame:
    messages = messages.copy()
    messages["state"] = messages.apply(_state_for_row, axis=1)
    return messages


ALL_STATES = [lbl for _, lbl in USER_STATE_PRIORITY] + ["user_neutral"] + \
             [lbl for _, lbl in ASTRO_STATE_PRIORITY] + ["astro_other"]


def transition_matrix(messages: pd.DataFrame) -> pd.DataFrame:
    labeled = label_states(messages)
    counts = pd.DataFrame(0, index=ALL_STATES, columns=ALL_STATES, dtype=float)
    for _, session_msgs in labeled.groupby("session_id"):
        session_msgs = session_msgs.sort_values("message_order")
        states = session_msgs["state"].tolist()
        for a, b in zip(states, states[1:]):
            counts.loc[a, b] += 1
    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums.replace(0, np.nan), axis=0)
    return probs.fillna(0.0)


def plot_transition_heatmap(matrix: pd.DataFrame, title: str, path):
    fig, ax = new_fig(figsize=(11, 9.5))
    im = ax.imshow(matrix.values, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=matrix.values.max() or 1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=75, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6.5)
    ax.set_xlabel("next state")
    ax.set_ylabel("state")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    save(fig, path)


def plot_diff_heatmap(diff: pd.DataFrame, title: str, path):
    fig, ax = new_fig(figsize=(11, 9.5))
    bound = max(abs(diff.values.min()), abs(diff.values.max()), 1e-6)
    im = ax.imshow(diff.values, cmap=DIVERGING, vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(diff.columns)))
    ax.set_xticklabels(diff.columns, rotation=75, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(diff.index)))
    ax.set_yticklabels(diff.index, fontsize=6.5)
    ax.set_xlabel("next state")
    ax.set_ylabel("state")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    save(fig, path)


def top_diff_cells(diff: pd.DataFrame, n=20) -> pd.DataFrame:
    stacked = diff.stack().rename("prob_diff").reset_index()
    stacked.columns = ["state", "next_state", "prob_diff"]
    stacked["abs_diff"] = stacked["prob_diff"].abs()
    return stacked.sort_values("abs_diff", ascending=False).head(n).drop(columns="abs_diff")


def run():
    group_frames = groups_mod.get_groups()
    matrices = {}

    for name, frames in group_frames.items():
        mat = transition_matrix(frames["messages"])
        matrices[name] = mat
        mat.to_csv(OUT_DIR / f"transition_matrix_{name}.csv")
        plot_transition_heatmap(mat, f"State transition probabilities: {name}",
                                 OUT_DIR / f"heatmap_transitions_{name}.png")

        for bucket in ["1", "2", "3+"]:
            sub = frames["messages"][frames["messages"]["session_bucket"] == bucket]
            if sub.empty:
                continue
            sub_mat = transition_matrix(sub)
            fname = f"transition_matrix_{name}_session_{bucket.replace('+','plus')}"
            sub_mat.to_csv(OUT_DIR / f"{fname}.csv")
            plot_transition_heatmap(sub_mat, f"State transitions: {name} (session {bucket})",
                                     OUT_DIR / f"heatmap_{fname}.png")

    baseline_mat = matrices["baseline_returned"]
    for sitara in ["Sitara-1", "Sitara-2"]:
        diff = matrices[sitara] - baseline_mat
        diff.to_csv(OUT_DIR / f"diff_{sitara}_minus_baseline_returned.csv")
        plot_diff_heatmap(diff, f"{sitara} minus baseline_returned (transition prob. diff)",
                           OUT_DIR / f"heatmap_diff_{sitara}_minus_baseline_returned.png")
        top = top_diff_cells(diff)
        top.to_csv(OUT_DIR / f"top_diff_cells_{sitara}_vs_baseline_returned.csv", index=False)

    print(f"Stage 4.3 Markov transition modeling written to {OUT_DIR}")
    return matrices


if __name__ == "__main__":
    run()
