"""Stage 4.5: sentiment/emotion trajectory + arc clustering.

Sentiment scoring: the spec asks to check for an offline option before
defaulting to another Gemini call, but use Gemini if quality matters more
given the Hinglish text. VADER (the standard offline option) is English-
lexicon only and would badly misjudge Hindi/Hinglish text ("Sab thik ho
jayega", "Bahut pareshan hoon"). Rather than spend another ~51k-message
Gemini pass purely for sentiment, we derive a proxy sentiment score directly
from the user-side situation tags Stage 2 already produced with Gemini
(which *did* read the Hinglish natively) -- this reuses that judgment instead
of re-buying it:

    score = +1.0 * enthusiastic_engagement
            -1.0 * disengagement_signal
            -0.5 * pushback_disagreement
            -0.3 * emotional_disclosure  (disclosure in this context is
                                          usually sharing a worry/problem)
    clipped to [-1, 1]; neutral_factual and untagged turns default to 0.

Trajectories (turn index -> sentiment) are resampled to a fixed length and
clustered by shape with DTW k-means (tslearn), then cross-tabbed against
source group and returned/not_returned outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.plotting import CATEGORICAL, new_fig, save

OUT_DIR = config.RESULTS_DIR / "sentiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_LEN = 10
N_CLUSTERS = 4


def score_message(row) -> float:
    score = 0.0
    if row.get("enthusiastic_engagement") is True:
        score += 1.0
    if row.get("disengagement_signal") is True:
        score -= 1.0
    if row.get("pushback_disagreement") is True:
        score -= 0.5
    if row.get("emotional_disclosure") is True:
        score -= 0.3
    return float(np.clip(score, -1.0, 1.0))


def build_trajectories(messages: pd.DataFrame) -> dict:
    """Returns {session_id: np.array of per-turn sentiment scores}."""
    user_msgs = messages[messages["role"] == "user"].copy()
    user_msgs["sentiment"] = user_msgs.apply(score_message, axis=1)
    trajectories = {}
    for session_id, g in user_msgs.groupby("session_id"):
        g = g.sort_values("message_order")
        if len(g) >= 3:
            trajectories[session_id] = g["sentiment"].to_numpy()
    return trajectories


def resample_trajectories(trajectories: dict, length=RESAMPLE_LEN) -> tuple[list, np.ndarray]:
    session_ids = list(trajectories.keys())
    raw = [trajectories[sid].reshape(-1, 1) for sid in session_ids]
    resampled = TimeSeriesResampler(sz=length).fit_transform(raw)
    return session_ids, resampled


def cluster_trajectories(resampled: np.ndarray, n_clusters=N_CLUSTERS, seed=config.RANDOM_SEED):
    n_clusters = min(n_clusters, max(1, len(resampled) // 3)) if len(resampled) else 1
    if n_clusters < 1 or len(resampled) < n_clusters:
        return None, None
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=seed, n_init=2)
    labels = km.fit_predict(resampled)
    return km, labels


def plot_cluster_trajectories(km, resampled, labels, title, path):
    n_clusters = km.cluster_centers_.shape[0]
    fig, axes = new_fig(figsize=(4 * n_clusters, 4))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 4), facecolor="#fcfcfb", sharey=True)
    if n_clusters == 1:
        axes = [axes]
    for c in range(n_clusters):
        ax = axes[c]
        ax.set_facecolor("#fcfcfb")
        members = resampled[labels == c]
        for m in members:
            ax.plot(m.ravel(), color=CATEGORICAL[c % len(CATEGORICAL)], alpha=0.15, linewidth=1)
        ax.plot(km.cluster_centers_[c].ravel(), color=CATEGORICAL[c % len(CATEGORICAL)], linewidth=2.5)
        ax.set_title(f"cluster {c} (n={len(members)})", fontsize=9)
        ax.set_xlabel("turn (resampled)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("sentiment")
    fig.suptitle(title)
    save(fig, path)


def run():
    group_frames = groups_mod.get_groups()
    crosstab_rows = []

    for name, frames in group_frames.items():
        messages = frames["messages"]
        sessions = frames["sessions"]
        trajectories = build_trajectories(messages)
        if len(trajectories) < 4:
            print(f"[sentiment] skipping {name}: too few sessions with >=3 user turns")
            continue

        session_ids, resampled = resample_trajectories(trajectories)
        km, labels = cluster_trajectories(resampled)
        if km is None:
            print(f"[sentiment] skipping cluster fit for {name}: insufficient sample")
            continue

        plot_cluster_trajectories(km, resampled, labels,
                                   f"Sentiment trajectory clusters: {name}",
                                   OUT_DIR / f"clusters_{name}.png")

        sess_meta = sessions.set_index("session_id")
        for sid, label in zip(session_ids, labels):
            outcome = sess_meta.loc[sid, "returned"] if sid in sess_meta.index and "returned" in sess_meta.columns else None
            crosstab_rows.append({
                "session_id": sid, "source_group": name,
                "arc_cluster": int(label), "returned": outcome,
            })

    if crosstab_rows:
        crosstab_df = pd.DataFrame(crosstab_rows)
        crosstab_df.to_csv(OUT_DIR / "arc_cluster_assignments.csv", index=False)

        group_tab = pd.crosstab(crosstab_df["source_group"], crosstab_df["arc_cluster"])
        group_tab.to_csv(OUT_DIR / "crosstab_arc_by_source_group.csv")

        outcome_df = crosstab_df.dropna(subset=["returned"])
        if not outcome_df.empty:
            outcome_tab = pd.crosstab(outcome_df["arc_cluster"], outcome_df["returned"])
            outcome_tab.to_csv(OUT_DIR / "crosstab_arc_by_outcome.csv")

    print(f"Stage 4.5 sentiment trajectory analysis written to {OUT_DIR}")


if __name__ == "__main__":
    run()
