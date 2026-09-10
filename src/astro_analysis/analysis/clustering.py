"""Stage 4.9: whole-conversation clustering by tag-sequence similarity.

Computes pairwise Levenshtein edit distance between each session's tag
sequence (states reused from the Markov module, encoded as single characters
so textdistance can compare them token-wise), clusters via hierarchical
clustering, and uses the resulting cluster labels to populate the
`archetype` field left null in Stage 2. Cross-tabs archetype membership
against source group and outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import textdistance
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.manifold import MDS

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.markov import ALL_STATES, label_states
from astro_analysis.analysis.plotting import CATEGORICAL, new_fig, save

OUT_DIR = config.RESULTS_DIR / "clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ARCHETYPES = 6
MAX_SESSIONS_FOR_DISTANCE_MATRIX = 400  # O(n^2) edit-distance; cap for tractability

STATE_TO_TOKEN = {state: chr(65 + i) for i, state in enumerate(ALL_STATES)}


def session_sequences(messages: pd.DataFrame) -> dict:
    labeled = label_states(messages)
    seqs = {}
    for sid, g in labeled.groupby("session_id"):
        g = g.sort_values("message_order")
        seqs[sid] = "".join(STATE_TO_TOKEN.get(s, "?") for s in g["state"])
    return seqs


def levenshtein_distance_matrix(sequences: dict, seed=config.RANDOM_SEED) -> tuple[list, np.ndarray]:
    session_ids = list(sequences.keys())
    rng = np.random.default_rng(seed)
    if len(session_ids) > MAX_SESSIONS_FOR_DISTANCE_MATRIX:
        session_ids = list(rng.choice(session_ids, MAX_SESSIONS_FOR_DISTANCE_MATRIX, replace=False))

    n = len(session_ids)
    dist = np.zeros((n, n))
    seqs = [sequences[sid] for sid in session_ids]
    for i in range(n):
        for j in range(i + 1, n):
            d = textdistance.levenshtein.distance(seqs[i], seqs[j])
            norm = d / max(len(seqs[i]), len(seqs[j]), 1)
            dist[i, j] = dist[j, i] = norm
    return session_ids, dist


def cluster_sessions(dist: np.ndarray, n_clusters=N_ARCHETYPES) -> tuple[np.ndarray, np.ndarray]:
    condensed = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=min(n_clusters, max(1, len(dist) - 1)), criterion="maxclust")
    return Z, labels


def plot_dendrogram(Z, session_ids, title, path):
    fig, ax = new_fig(figsize=(max(10, len(session_ids) * 0.08), 6))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0, above_threshold_color=CATEGORICAL[0])
    ax.set_xlabel("session")
    ax.set_ylabel("distance")
    ax.set_title(title)
    save(fig, path)


def plot_mds_projection(dist: np.ndarray, labels: np.ndarray, color_by: list, title: str, path, seed=config.RANDOM_SEED):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, normalized_stress="auto")
    coords = mds.fit_transform(dist)
    fig, ax = new_fig()
    categories = sorted(set(color_by))
    for i, cat in enumerate(categories):
        mask = [c == cat for c in color_by]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, color=CATEGORICAL[i % len(CATEGORICAL)], label=str(cat))
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7, markerscale=1.2)
    save(fig, path)


def run():
    group_frames = groups_mod.get_groups()
    all_messages = pd.concat([f["messages"].assign(source_group=name) for name, f in group_frames.items()],
                              ignore_index=True)
    all_sessions = pd.concat([f["sessions"].assign(source_group=name) for name, f in group_frames.items()],
                              ignore_index=True)

    sequences = session_sequences(all_messages)
    session_ids, dist = levenshtein_distance_matrix(sequences)
    Z, labels = cluster_sessions(dist)

    plot_dendrogram(Z, session_ids, "Session tag-sequence clustering (all groups)",
                     OUT_DIR / "dendrogram_all_groups.png")

    meta = all_sessions.set_index("session_id")
    group_by_session = [meta.loc[sid, "source_group"] if sid in meta.index else "unknown" for sid in session_ids]
    outcome_by_session = [
        (meta.loc[sid, "returned"] if sid in meta.index and "returned" in meta.columns else None)
        for sid in session_ids
    ]

    plot_mds_projection(dist, labels, group_by_session,
                         "MDS projection of session tag-sequences, colored by source group",
                         OUT_DIR / "mds_by_source_group.png")
    plot_mds_projection(dist, labels, [str(o) for o in outcome_by_session],
                         "MDS projection of session tag-sequences, colored by outcome",
                         OUT_DIR / "mds_by_outcome.png")

    archetype_df = pd.DataFrame({
        "session_id": session_ids,
        "archetype": [f"archetype_{l}" for l in labels],
        "source_group": group_by_session,
        "returned": outcome_by_session,
    })
    archetype_df.to_csv(OUT_DIR / "archetype_assignments.csv", index=False)

    crosstab_group = pd.crosstab(archetype_df["archetype"], archetype_df["source_group"])
    crosstab_group.to_csv(OUT_DIR / "crosstab_archetype_by_source_group.csv")

    outcome_df = archetype_df.dropna(subset=["returned"])
    if not outcome_df.empty:
        crosstab_outcome = pd.crosstab(outcome_df["archetype"], outcome_df["returned"])
        crosstab_outcome.to_csv(OUT_DIR / "crosstab_archetype_by_outcome.csv")

    # persist archetype labels back onto the sessions_tagged table, per Stage 2's
    # spec that `archetype` gets filled in here rather than by the tagging LLM
    sessions_tagged = pd.read_parquet(config.SESSIONS_TAGGED_PATH)
    archetype_map = archetype_df.set_index("session_id")["archetype"]
    sessions_tagged["archetype"] = sessions_tagged["session_id"].map(archetype_map).combine_first(
        sessions_tagged["archetype"])
    sessions_tagged.to_parquet(config.SESSIONS_TAGGED_PATH, index=False)

    print(f"Stage 4.9 clustering written to {OUT_DIR}; archetype labels persisted "
          f"to {config.SESSIONS_TAGGED_PATH}")


if __name__ == "__main__":
    run()
