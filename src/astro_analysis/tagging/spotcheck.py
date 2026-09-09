"""Stage 2 validation: sample N tagged messages with context for manual review."""
from __future__ import annotations

import pandas as pd

from astro_analysis import config
from astro_analysis.tagging.tagger import ALL_MESSAGE_FIELDS


def build_review_sample(messages_tagged: pd.DataFrame, n: int = 50, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    tagged_mask = messages_tagged[ALL_MESSAGE_FIELDS].notna().any(axis=1)
    candidates = messages_tagged[tagged_mask]
    n = min(n, len(candidates))
    sample = candidates.sample(n=n, random_state=seed).sort_values(["session_id", "message_order"])

    rows = []
    for _, row in sample.iterrows():
        session_msgs = messages_tagged[messages_tagged["session_id"] == row["session_id"]].sort_values("message_order")
        context = session_msgs[session_msgs["message_order"] < row["message_order"]].tail(3)
        context_str = " | ".join(f"[{c['role']}] {c['message']}" for _, c in context.iterrows())
        rows.append({
            "session_id": row["session_id"],
            "source_folder": row["source_folder"],
            "session_number": row["session_number"],
            "message_order": row["message_order"],
            "role": row["role"],
            "context_prior_turns": context_str,
            "message": row["message"],
            **{f: row[f] for f in ALL_MESSAGE_FIELDS},
            "reviewer_verdict": "",  # for manual fill-in: correct / incorrect / unsure
            "reviewer_notes": "",
        })
    return pd.DataFrame(rows)


def run(n: int = 50, input_path=None, output_path=None):
    input_path = input_path or config.MESSAGES_TAGGED_PATH
    output_path = output_path or (config.RESULTS_DIR / "tagging_spotcheck_sample.csv")
    messages_tagged = pd.read_parquet(input_path)
    sample_df = build_review_sample(messages_tagged, n=n)
    sample_df.to_csv(output_path, index=False)
    print(f"Wrote {len(sample_df)} rows for manual review -> {output_path}")
    return sample_df


if __name__ == "__main__":
    run()
