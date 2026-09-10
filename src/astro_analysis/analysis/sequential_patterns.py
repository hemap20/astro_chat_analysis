"""Stage 4.2: sequential pattern mining (PrefixSpan) over per-message states
(reusing the same state labels as the Markov module), separately per source
group and per session_number bucket, to surface recurring strategy
sequences."""
from __future__ import annotations

import pandas as pd
from prefixspan import PrefixSpan

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.markov import label_states
from astro_analysis.analysis.plotting import CATEGORICAL, new_fig, save

OUT_DIR = config.RESULTS_DIR / "sequential_patterns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PATTERN_LEN = 2
TOP_N = 15


def build_sequences(messages: pd.DataFrame) -> list[list[str]]:
    labeled = label_states(messages)
    sequences = []
    for _, session_msgs in labeled.groupby("session_id"):
        seq = session_msgs.sort_values("message_order")["state"].tolist()
        if len(seq) >= MIN_PATTERN_LEN:
            sequences.append(seq)
    return sequences


def mine_top_patterns(sequences: list[list[str]], top_n=TOP_N, min_len=MIN_PATTERN_LEN) -> pd.DataFrame:
    if not sequences:
        return pd.DataFrame(columns=["pattern", "support", "support_frac", "length"])
    ps = PrefixSpan(sequences)
    ps.minlen = min_len
    results = ps.topk(top_n * 3)  # over-fetch, then filter short/trivial ones
    rows = []
    for support, pattern in results:
        if len(pattern) < min_len:
            continue
        rows.append({
            "pattern": " -> ".join(pattern),
            "support": support,
            "support_frac": support / len(sequences),
            "length": len(pattern),
        })
    df = pd.DataFrame(rows).sort_values("support", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def plot_top_patterns(df: pd.DataFrame, title: str, path):
    if df.empty:
        return
    fig, ax = new_fig(figsize=(10, max(4, 0.4 * len(df))))
    y = range(len(df))
    ax.barh(y, df["support_frac"], color=CATEGORICAL[0])
    ax.set_yticks(y)
    ax.set_yticklabels(df["pattern"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of sessions containing this sub-sequence")
    ax.set_title(title)
    save(fig, path)


def plot_example_timeline(messages: pd.DataFrame, session_id: str, title: str, path):
    labeled = label_states(messages[messages["session_id"] == session_id]).sort_values("message_order")
    if labeled.empty:
        return
    states = labeled["state"].tolist()
    unique_states = sorted(set(states))
    state_to_y = {s: i for i, s in enumerate(unique_states)}
    colors = [CATEGORICAL[0] if s.startswith("user") else CATEGORICAL[1] for s in states]

    fig, ax = new_fig(figsize=(11, max(3, 0.35 * len(unique_states))))
    ax.scatter(range(len(states)), [state_to_y[s] for s in states], c=colors, s=30)
    ax.plot(range(len(states)), [state_to_y[s] for s in states], color="#c3c2b7", linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(unique_states)))
    ax.set_yticklabels(unique_states, fontsize=7)
    ax.set_xlabel("turn index")
    ax.set_title(title)
    save(fig, path)


def run():
    group_frames = groups_mod.get_groups()

    for name, frames in group_frames.items():
        messages = frames["messages"]
        sequences = build_sequences(messages)
        top = mine_top_patterns(sequences)
        top.to_csv(OUT_DIR / f"top_patterns_{name}.csv", index=False)
        plot_top_patterns(top, f"Top recurring tag sequences: {name}",
                           OUT_DIR / f"bar_top_patterns_{name}.png")

        for bucket in ["1", "2", "3+"]:
            sub_msgs = messages[messages["session_bucket"] == bucket]
            sub_seqs = build_sequences(sub_msgs)
            sub_top = mine_top_patterns(sub_seqs)
            fname = f"top_patterns_{name}_session_{bucket.replace('+','plus')}"
            sub_top.to_csv(OUT_DIR / f"{fname}.csv", index=False)
            plot_top_patterns(sub_top, f"Top sequences: {name} (session {bucket})",
                               OUT_DIR / f"bar_{fname}.png")

        # illustrative example: longest session in this group, as a timeline
        if not messages.empty:
            longest_session = messages.groupby("session_id").size().idxmax()
            plot_example_timeline(messages, longest_session,
                                   f"Example session timeline: {name}",
                                   OUT_DIR / f"timeline_example_{name}.png")

    print(f"Stage 4.2 sequential pattern mining written to {OUT_DIR}")


if __name__ == "__main__":
    run()
