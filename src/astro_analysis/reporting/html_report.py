"""Output & reporting layer: a single static HTML report walking every Stage 4
analysis with its charts and a short auto-generated caption of the headline
numbers, for sharing with a non-technical product team.
"""
from __future__ import annotations

import html as html_escape
from pathlib import Path

import pandas as pd

from astro_analysis import config

RESULTS_DIR = config.RESULTS_DIR
REPORT_PATH = RESULTS_DIR / "report.html"

SECTION_ORDER = [
    ("tag_rates", "Tag Rate & Combination Comparison"),
    ("sequential_patterns", "Sequential Pattern Mining"),
    ("markov", "Markov Transition Modeling"),
    ("survival", "Survival Analysis"),
    ("sentiment", "Sentiment / Emotion Trajectories"),
    ("paradise", "PARADISE-style Regression"),
    ("diff_in_diff", "Difference-in-Differences (Scaffold)"),
    ("process_mining", "Process Mining"),
    ("clustering", "Tag-Sequence Clustering / Archetypes"),
]


def _caption_tag_rates(folder: Path) -> str:
    path = folder / "pairwise_baseline_returned_vs_baseline_not_returned.csv"
    if not path.exists():
        return "Tag rate comparison across source groups and session-number buckets."
    df = pd.read_csv(path, index_col=0)
    diffs = (df.loc["baseline_returned"] - df.loc["baseline_not_returned"]).dropna()
    diffs = diffs[~diffs.index.str.startswith("n_")]
    if diffs.empty:
        return "Tag rate comparison across source groups."
    top = diffs.abs().sort_values(ascending=False).head(3)
    parts = [f"{tag} ({diffs[tag]:+.2f})" for tag in top.index]
    return ("Biggest returned-vs-not-returned gaps: " + ", ".join(parts) +
            ". Positive = more common among users who returned.")


def _caption_markov(folder: Path) -> str:
    path = folder / "top_diff_cells_Sitara-1_vs_baseline_returned.csv"
    if not path.exists():
        return "Per-group state transition matrices, plus Sitara vs. baseline_returned diffs."
    df = pd.read_csv(path)
    if df.empty:
        return "Per-group state transition matrices."
    row = df.iloc[0]
    return (f"Largest Sitara-1 vs. baseline_returned transition shift: "
            f"{row['state']} -> {row['next_state']} ({row['prob_diff']:+.2f} probability).")


def _caption_survival(folder: Path) -> str:
    path = folder / "hazard_ratios_combined_disengagement.csv"
    if not path.exists():
        return "Cox proportional-hazards models for disengagement and positive continuation, per group."
    df = pd.read_csv(path, index_col=0)
    if df.empty or "exp(coef)" not in df.columns:
        return "Cox proportional-hazards models for disengagement and positive continuation."
    protective = df.sort_values("exp(coef)").iloc[0]
    risky = df.sort_values("exp(coef)", ascending=False).iloc[0]
    return (f"Combined model: {protective.name} most protective against disengagement "
            f"(HR={protective['exp(coef)']:.2f}); {risky.name} most associated with "
            f"higher disengagement hazard (HR={risky['exp(coef)']:.2f}).")


def _caption_sentiment(folder: Path) -> str:
    path = folder / "crosstab_arc_by_source_group.csv"
    if not path.exists():
        return "Sentiment trajectory shapes, clustered by DTW, per source group."
    df = pd.read_csv(path, index_col=0)
    return (f"{df.shape[1]} recurring emotional-arc shapes identified across "
            f"{df.values.sum()} sessions; distribution varies by source group (see table).")


def _caption_paradise(folder: Path) -> str:
    path = folder / "coefficients_combined_returned_next_session.csv"
    if not path.exists():
        return "Regression of outcome signals against tag rates and trajectory features."
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        return "Regression of outcome signals against tag rates and trajectory features."
    top = df["coefficient"].abs().sort_values(ascending=False).index[0]
    coef = df.loc[top, "coefficient"]
    direction = "increases" if coef > 0 else "decreases"
    return f"Combined model: {top} most strongly {direction} predicted odds of returning next session."


def _caption_diff_in_diff(folder: Path) -> str:
    return ("Illustrative Sitara-1 vs. Sitara-2 demo only -- versions were not run "
            "concurrently, so no causal estimate is claimed. See CSV for the raw "
            "post-only difference and the scaffold's `run_did` for real future use.")


def _caption_process_mining(folder: Path) -> str:
    return ("Directly-follows graphs (top-10 most frequent states) per source group. "
            "PNG process maps require Graphviz; edge frequency tables are always available as CSV.")


def _caption_clustering(folder: Path) -> str:
    path = folder / "crosstab_archetype_by_source_group.csv"
    if not path.exists():
        return "Sessions clustered by tag-sequence edit distance into conversational archetypes."
    df = pd.read_csv(path, index_col=0)
    return (f"{len(df)} archetypes identified from tag-sequence similarity across "
            f"{df.values.sum()} sessions; see dendrogram and MDS projection for structure.")


def _caption_sequential_patterns(folder: Path) -> str:
    path = folder / "top_patterns_baseline_returned.csv"
    if not path.exists():
        return "Most frequent recurring tag sub-sequences per source group (PrefixSpan)."
    df = pd.read_csv(path)
    if df.empty:
        return "Most frequent recurring tag sub-sequences per source group."
    top = df.iloc[0]
    return (f"Most common baseline_returned sub-sequence: \"{top['pattern']}\" "
            f"(in {top['support_frac']:.0%} of sessions).")


CAPTION_FNS = {
    "tag_rates": _caption_tag_rates,
    "markov": _caption_markov,
    "survival": _caption_survival,
    "sentiment": _caption_sentiment,
    "paradise": _caption_paradise,
    "diff_in_diff": _caption_diff_in_diff,
    "process_mining": _caption_process_mining,
    "clustering": _caption_clustering,
    "sequential_patterns": _caption_sequential_patterns,
}

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Astro Analysis - Stage 4 Report</title>
<style>
body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1100px;
       margin: 40px auto; padding: 0 20px; color: #0b0b0b; background: #fcfcfb; }
h1 { font-size: 28px; }
h2 { font-size: 20px; margin-top: 56px; border-bottom: 2px solid #e3e2dd; padding-bottom: 8px; }
.caption { color: #52514e; font-size: 14px; margin: 8px 0 20px; max-width: 800px; }
.chart { margin-bottom: 28px; }
.chart img { max-width: 100%; border: 1px solid #e3e2dd; border-radius: 6px; }
.chart-title { font-size: 13px; color: #52514e; margin-bottom: 6px; font-family: monospace; }
nav { background: #fff; border: 1px solid #e3e2dd; border-radius: 8px; padding: 16px 20px; margin-bottom: 32px; }
nav a { display: block; padding: 3px 0; color: #2a78d6; text-decoration: none; font-size: 14px; }
nav a:hover { text-decoration: underline; }
.limitations { background: #fff8ee; border: 1px solid #eda100; border-radius: 8px; padding: 16px 20px; margin-top: 40px; font-size: 14px; }
</style></head><body>
"""

LIMITATIONS_HTML = """
<div class="limitations">
<strong>Known limitations to read this report with:</strong>
<ul>
<li>baseline_not_returned relies on a small pool of genuinely single-session Human_single users; treat comparisons involving it as directional, not definitive, until more such users are added.</li>
<li>1 of 1,008 sessions (a pathological Human_Astro conversation) failed automated tagging after repeated retries and is excluded from all analyses.</li>
<li>The difference-in-differences module is a reusable scaffold; the Sitara-1 vs. Sitara-2 run shown is illustrative only (the versions were not run concurrently) and is not a causal estimate.</li>
<li>Survival analysis and PARADISE regression use static, session-level tag rates as covariates (not turn-by-turn time-varying covariates), a tractability simplification given sample size.</li>
<li>Sentiment scores are a proxy derived from Stage 2's Gemini-tagged situation tags (enthusiastic_engagement, disengagement_signal, pushback_disagreement, emotional_disclosure), not a separately re-scored sentiment model -- see astro_analysis.analysis.sentiment docstring.</li>
</ul>
</div>
"""


def build_report():
    sections_html = []
    nav_html = []

    for folder_name, title in SECTION_ORDER:
        folder = RESULTS_DIR / folder_name
        if not folder.exists():
            continue
        pngs = sorted(folder.glob("*.png"))
        if not pngs:
            continue

        anchor = folder_name
        nav_html.append(f'<a href="#{anchor}">{html_escape.escape(title)}</a>')

        caption_fn = CAPTION_FNS.get(folder_name)
        caption = caption_fn(folder) if caption_fn else ""

        charts_html = []
        for png in pngs:
            rel = png.relative_to(RESULTS_DIR)
            charts_html.append(
                f'<div class="chart"><div class="chart-title">{html_escape.escape(png.stem)}</div>'
                f'<img src="{rel.as_posix()}" loading="lazy"></div>'
            )

        sections_html.append(
            f'<h2 id="{anchor}">{html_escape.escape(title)}</h2>'
            f'<p class="caption">{html_escape.escape(caption)}</p>'
            + "".join(charts_html)
        )

    body = (
        HTML_HEAD
        + "<h1>Astro Analysis: AI vs. Human Astrologer Conversational Strategies</h1>"
        + '<p class="caption">Auto-generated Stage 4 report. Each section\'s caption states the headline '
          "numbers; full detail is in the accompanying CSVs in results/&lt;section&gt;/.</p>"
        + f"<nav>{''.join(nav_html)}</nav>"
        + "".join(sections_html)
        + LIMITATIONS_HTML
        + "</body></html>"
    )
    REPORT_PATH.write_text(body)
    print(f"Report written to {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    build_report()
