"""Stage 4.1: tag rate & combination comparison.

Computes, for every pair of groups worth comparing (baseline_returned vs.
baseline_not_returned, Sitara-1 vs. Sitara-2, each Sitara vs.
baseline_returned) and for the pooled view plus each session_number bucket
(1 / 2 / 3+):
  - boolean tag rates (astrologer tags among astrologer messages, user tags
    among user messages, session-level tags among sessions)
  - categorical tag value distributions (claim_certainty, specificity,
    repair_after_pushback, ending_type)
  - pairwise co-occurrence of boolean astrologer tags (for the heatmap)

Outputs CSVs + PNGs into results/tag_rates/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.plotting import CATEGORICAL, SEQUENTIAL_BLUE, new_fig, save, style_axes

OUT_DIR = config.RESULTS_DIR / "tag_rates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOL_ASTRO_TAGS = [
    "concrete_dated_prediction", "specific_open_thread_named",
    "callback_prior_session", "callback_within_session",
    "validation", "self_correction", "reframe",
]
BOOL_USER_TAGS = [
    "emotional_disclosure", "pushback_disagreement", "enthusiastic_engagement",
    "disengagement_signal", "neutral_factual",
]
CATEGORICAL_ASTRO_TAGS = ["claim_certainty", "specificity", "repair_after_pushback"]


def _bool_rate(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.astype(bool).mean())


def compute_tag_rates(messages: pd.DataFrame) -> pd.Series:
    """One row of tag-rate stats for a slice of the messages table."""
    astro = messages[messages["role"] == "astrologer"]
    user = messages[messages["role"] == "user"]

    rates = {}
    for tag in BOOL_ASTRO_TAGS:
        rates[tag] = _bool_rate(astro[tag]) if tag in astro else np.nan
    for tag in BOOL_USER_TAGS:
        rates[tag] = _bool_rate(user[tag]) if tag in user else np.nan

    for tag in CATEGORICAL_ASTRO_TAGS:
        if tag not in astro:
            continue
        vc = astro[tag].dropna()
        vc = vc[vc != "not_applicable"]
        total = len(vc)
        for val, count in vc.value_counts().items():
            rates[f"{tag}={val}"] = count / total if total else np.nan

    rates["n_astro_messages"] = len(astro)
    rates["n_user_messages"] = len(user)
    return pd.Series(rates)


def compute_session_level_rates(sessions: pd.DataFrame) -> pd.Series:
    out = {}
    if "wall_hit" in sessions:
        out["wall_hit"] = _bool_rate(sessions["wall_hit"])
    if "ending_type" in sessions:
        vc = sessions["ending_type"].dropna()
        total = len(vc)
        for val, count in vc.value_counts().items():
            out[f"ending_type={val}"] = count / total if total else np.nan
    out["n_sessions"] = len(sessions)
    return pd.Series(out)


def build_rate_table(group_frames: dict, bucket: str | None = None) -> pd.DataFrame:
    rows = {}
    for name, frames in group_frames.items():
        messages = frames["messages"]
        sessions = frames["sessions"]
        if bucket is not None:
            messages = messages[messages["session_bucket"] == bucket]
            sessions = sessions[sessions["session_bucket"] == bucket]
        msg_rates = compute_tag_rates(messages)
        sess_rates = compute_session_level_rates(sessions)
        rows[name] = pd.concat([msg_rates, sess_rates])
    return pd.DataFrame(rows).T


def cooccurrence_matrix(messages: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    astro = messages[messages["role"] == "astrologer"][tags].apply(
        lambda c: c.astype("boolean"))
    astro = astro.dropna(how="all")
    astro = astro.fillna(False).astype(bool).astype(int)
    if astro.empty:
        return pd.DataFrame(np.nan, index=tags, columns=tags)
    return astro.corr()


def plot_grouped_bars(rate_table: pd.DataFrame, tags: list[str], title: str, path):
    fig, ax = new_fig(figsize=(max(9, len(tags) * 0.9), 5.5))
    n_groups = len(rate_table.index)
    width = 0.8 / max(n_groups, 1)
    x = np.arange(len(tags))
    for i, (group_name, row) in enumerate(rate_table.iterrows()):
        vals = [row.get(t, np.nan) for t in tags]
        ax.bar(x + i * width, vals, width=width, label=group_name,
               color=CATEGORICAL[i % len(CATEGORICAL)])
    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    save(fig, path)


def plot_cooccurrence_heatmap(corr: pd.DataFrame, title: str, path):
    fig, ax = new_fig(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    save(fig, path)


def run():
    group_frames = groups_mod.get_groups()

    # 1. pooled tag rate table across all four groups
    pooled = build_rate_table(group_frames)
    pooled.to_csv(OUT_DIR / "tag_rates_pooled_all_groups.csv")
    plot_grouped_bars(pooled, BOOL_ASTRO_TAGS,
                       "Astrologer tag rates (pooled, all sessions)",
                       OUT_DIR / "bar_astro_tags_pooled.png")
    plot_grouped_bars(pooled, BOOL_USER_TAGS,
                       "User situation tag rates (pooled, all sessions)",
                       OUT_DIR / "bar_user_tags_pooled.png")

    # 2. by session_number bucket
    for bucket in ["1", "2", "3+"]:
        table = build_rate_table(group_frames, bucket=bucket)
        table.to_csv(OUT_DIR / f"tag_rates_session_bucket_{bucket.replace('+','plus')}.csv")
        plot_grouped_bars(table, BOOL_ASTRO_TAGS,
                           f"Astrologer tag rates (session {bucket})",
                           OUT_DIR / f"bar_astro_tags_session_{bucket.replace('+','plus')}.png")

    # 3. specific pairwise comparisons called out in the spec
    pairwise_specs = [
        ("baseline_returned", "baseline_not_returned"),
        ("Sitara-1", "Sitara-2"),
        ("Sitara-1", "baseline_returned"),
        ("Sitara-2", "baseline_returned"),
    ]
    for a, b in pairwise_specs:
        sub_table = pooled.loc[[a, b]]
        fname = f"pairwise_{a}_vs_{b}".replace(" ", "_")
        sub_table.to_csv(OUT_DIR / f"{fname}.csv")
        plot_grouped_bars(sub_table, BOOL_ASTRO_TAGS + BOOL_USER_TAGS,
                           f"{a} vs {b}: tag rates",
                           OUT_DIR / f"{fname}.png")

    # 4. co-occurrence heatmaps per group (astrologer bool tags)
    for name, frames in group_frames.items():
        corr = cooccurrence_matrix(frames["messages"], BOOL_ASTRO_TAGS)
        corr.to_csv(OUT_DIR / f"cooccurrence_{name}.csv")
        plot_cooccurrence_heatmap(corr, f"Astrologer tag co-occurrence: {name}",
                                   OUT_DIR / f"heatmap_cooccurrence_{name}.png")

    print(f"Stage 4.1 tag rate comparison written to {OUT_DIR}")
    return pooled


if __name__ == "__main__":
    run()
