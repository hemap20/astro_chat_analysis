"""Stage 4.4: bidirectional survival analysis (lifelines).

Unit of analysis is the SESSION. For each session we define two time-to-event
outcomes, both measured in turn (message) index within the session:

  disengagement: duration = turn index of the first user disengagement_signal
      in the session (event=1), or the session's total turn count if it never
      occurs (event=0, censored). "Session end without resolution"
      (ending_type != resolved_closed) with no explicit disengagement turn is
      also coded as an event at the final turn, since the conversation still
      ended badly.
  positive_continuation: duration = turn index of the first user
      enthusiastic_engagement (event=1), else session length (event=0,
      censored). A session that runs longer than this source group's median
      length with no disengagement is also coded as an event (the session
      "extended"), per the spec's "or session extending" clause.

Covariates are session-level astrologer tag RATES (fraction of astrologer
messages in the session where the tag fired) -- static per session, not
time-varying, to keep the Cox model tractable at this sample size. This is a
simplification worth stating plainly: it answers "sessions with more of this
behavior tend to end well/badly," not a turn-by-turn causal claim.

Hazard ratios are exp(coef): >1 means the tag is associated with the event
happening SOONER (higher hazard); for disengagement that's bad, for
positive_continuation that's good. Fit separately per source group (Stage 4
requirement) plus a combined pooled fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.plotting import CATEGORICAL, new_fig, save

OUT_DIR = config.RESULTS_DIR / "survival"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COVARIATE_TAGS = [
    "concrete_dated_prediction", "specific_open_thread_named",
    "callback_prior_session", "callback_within_session",
    "validation", "self_correction", "reframe",
]


def _session_tag_rates(session_msgs: pd.DataFrame) -> dict:
    astro = session_msgs[session_msgs["role"] == "astrologer"]
    rates = {}
    for tag in COVARIATE_TAGS:
        s = astro[tag].dropna()
        rates[tag] = float(s.astype(bool).mean()) if len(s) else 0.0
    return rates


def build_session_survival_table(messages: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    median_len = messages.groupby("session_id").size().median()

    for session_id, session_msgs in messages.groupby("session_id"):
        session_msgs = session_msgs.sort_values("message_order")
        n_turns = len(session_msgs)
        if n_turns < 2:
            continue
        sess_meta = sessions.loc[sessions["session_id"] == session_id]
        if sess_meta.empty:
            continue
        ending_type = sess_meta.iloc[0].get("ending_type")

        user_msgs = session_msgs[session_msgs["role"] == "user"].reset_index(drop=True)

        disengage_hits = user_msgs.index[user_msgs["disengagement_signal"] == True]  # noqa: E712
        if len(disengage_hits):
            dis_duration = int(user_msgs.loc[disengage_hits[0], "message_order"])
            dis_event = 1
        else:
            dis_duration = n_turns
            dis_event = 1 if ending_type not in (None, "resolved_closed") else 0

        enthu_hits = user_msgs.index[user_msgs["enthusiastic_engagement"] == True]  # noqa: E712
        if len(enthu_hits):
            pos_duration = int(user_msgs.loc[enthu_hits[0], "message_order"])
            pos_event = 1
        else:
            pos_duration = n_turns
            pos_event = 1 if n_turns > median_len and dis_event == 0 else 0

        row = {"session_id": session_id, "n_turns": n_turns,
               "disengage_duration": max(dis_duration, 1), "disengage_event": dis_event,
               "positive_duration": max(pos_duration, 1), "positive_event": pos_event}
        row.update(_session_tag_rates(session_msgs))
        rows.append(row)

    return pd.DataFrame(rows)


def fit_cox(table: pd.DataFrame, duration_col: str, event_col: str) -> CoxPHFitter | None:
    cols = [duration_col, event_col] + COVARIATE_TAGS
    fit_df = table[cols].dropna()
    if len(fit_df) < 15 or fit_df[event_col].sum() < 5:
        return None
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(fit_df, duration_col=duration_col, event_col=event_col)
    except Exception:
        return None
    return cph


def plot_km_curves(table: pd.DataFrame, duration_col: str, event_col: str,
                    tag: str, title: str, path):
    fig, ax = new_fig()
    kmf = KaplanMeierFitter()
    for i, present in enumerate([True, False]):
        mask = (table[tag] > 0) if present else (table[tag] == 0)
        sub = table[mask]
        if sub.empty:
            continue
        kmf.fit(sub[duration_col], sub[event_col], label=f"{tag}={'present' if present else 'absent'}")
        kmf.plot_survival_function(ax=ax, color=CATEGORICAL[i], ci_show=True)
    ax.set_xlabel("turn index")
    ax.set_ylabel("survival probability")
    ax.set_title(title)
    save(fig, path)


def plot_forest(cph: CoxPHFitter, title: str, path):
    summary = cph.summary.sort_values("exp(coef)")
    fig, ax = new_fig(figsize=(8, 0.5 * len(summary) + 2))
    y = np.arange(len(summary))
    ax.errorbar(summary["exp(coef)"], y,
                xerr=[summary["exp(coef)"] - summary["exp(coef) lower 95%"],
                      summary["exp(coef) upper 95%"] - summary["exp(coef)"]],
                fmt="o", color=CATEGORICAL[0], ecolor=CATEGORICAL[0], capsize=3)
    ax.axvline(1.0, color="#52514e", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index, fontsize=8)
    ax.set_xlabel("hazard ratio (exp(coef), log scale)")
    ax.set_title(title)
    save(fig, path)


def run():
    group_frames = groups_mod.get_groups()
    all_tables = {}

    for name, frames in group_frames.items():
        table = build_session_survival_table(frames["messages"], frames["sessions"])
        all_tables[name] = table
        table.to_csv(OUT_DIR / f"survival_table_{name}.csv", index=False)

        for label, dur, ev in [("disengagement", "disengage_duration", "disengage_event"),
                                ("positive_continuation", "positive_duration", "positive_event")]:
            cph = fit_cox(table, dur, ev)
            if cph is None:
                print(f"[survival] skipping {name}/{label}: insufficient events/sample")
                continue
            cph.summary.to_csv(OUT_DIR / f"hazard_ratios_{name}_{label}.csv")
            plot_forest(cph, f"{name}: hazard ratios ({label})",
                        OUT_DIR / f"forest_{name}_{label}.png")

            top_tag = cph.summary["exp(coef)"].abs().sort_values(ascending=False).index[0] \
                if len(cph.summary) else COVARIATE_TAGS[0]
            plot_km_curves(table, dur, ev, top_tag,
                           f"{name}: survival by {top_tag} ({label})",
                           OUT_DIR / f"km_{name}_{label}_{top_tag}.png")

    combined = pd.concat(
        [t.assign(source_group=name) for name, t in all_tables.items()], ignore_index=True
    )
    combined.to_csv(OUT_DIR / "survival_table_combined.csv", index=False)
    for label, dur, ev in [("disengagement", "disengage_duration", "disengage_event"),
                            ("positive_continuation", "positive_duration", "positive_event")]:
        cph = fit_cox(combined, dur, ev)
        if cph is not None:
            cph.summary.to_csv(OUT_DIR / f"hazard_ratios_combined_{label}.csv")
            plot_forest(cph, f"Combined: hazard ratios ({label})",
                        OUT_DIR / f"forest_combined_{label}.png")

    print(f"Stage 4.4 survival analysis written to {OUT_DIR}")
    return all_tables


if __name__ == "__main__":
    run()
