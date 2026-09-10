"""Stage 4.7: difference-in-differences scaffold.

`run_did` is a reusable, general-purpose function: given any sessions table,
a `group_col` (treated/control membership), a `date_col`, a `cutoff` date,
and a `metric_col`, it computes the standard 2x2 DiD estimate. This is built
now so a real rollout split (once one exists) can be plugged straight in.

The module also runs one illustrative example: Sitara-1 vs Sitara-2, treating
the Sitara-2 launch date as the "treatment" date. This is a placeholder demo
ONLY -- the two versions were never run concurrently, so there is no real
control group and this cannot support a causal claim. It exists to prove the
scaffold works end-to-end, not to estimate a real effect.
"""
from __future__ import annotations

import pandas as pd

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.plotting import CATEGORICAL, new_fig, save

OUT_DIR = config.RESULTS_DIR / "diff_in_diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_did(sessions: pd.DataFrame, group_col: str, treated_value, control_value,
            date_col: str, cutoff, metric_col: str) -> dict:
    """Standard DiD: (treated_post - treated_pre) - (control_post - control_pre)."""
    df = sessions.copy()
    df["_period"] = (pd.to_datetime(df[date_col]) >= pd.Timestamp(cutoff)).map({True: "post", False: "pre"})
    df = df[df[group_col].isin([treated_value, control_value])]

    means = df.groupby([group_col, "_period"])[metric_col].mean().unstack("_period")
    treated_pre = means.loc[treated_value, "pre"] if "pre" in means.columns else float("nan")
    treated_post = means.loc[treated_value, "post"] if "post" in means.columns else float("nan")
    control_pre = means.loc[control_value, "pre"] if "pre" in means.columns else float("nan")
    control_post = means.loc[control_value, "post"] if "post" in means.columns else float("nan")

    did_estimate = (treated_post - treated_pre) - (control_post - control_pre)

    return {
        "treated_pre": treated_pre, "treated_post": treated_post,
        "control_pre": control_pre, "control_post": control_post,
        "did_estimate": did_estimate,
        "naive_post_only_diff": treated_post - control_post,
        "means_table": means,
    }


def plot_parallel_trends(sessions: pd.DataFrame, group_col: str, treated_value, control_value,
                          date_col: str, cutoff, metric_col: str, title: str, path,
                          freq="W"):
    df = sessions.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_period_bucket"] = df[date_col].dt.to_period(freq).dt.start_time
    trend = df[df[group_col].isin([treated_value, control_value])].groupby(
        ["_period_bucket", group_col])[metric_col].mean().reset_index()

    fig, ax = new_fig()
    for i, (grp, sub) in enumerate(trend.groupby(group_col)):
        sub = sub.sort_values("_period_bucket")
        ax.plot(sub["_period_bucket"], sub[metric_col], marker="o", markersize=3,
                color=CATEGORICAL[i], label=str(grp))
    ax.axvline(pd.Timestamp(cutoff), color="#52514e", linestyle="--", linewidth=1, label="treatment cutoff")
    ax.set_xlabel("date")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    save(fig, path)


def run():
    group_frames = groups_mod.get_groups()
    sitara1 = group_frames["Sitara-1"]["sessions"].assign(version="Sitara-1")
    sitara2 = group_frames["Sitara-2"]["sessions"].assign(version="Sitara-2")
    combined = pd.concat([sitara1, sitara2], ignore_index=True)
    combined["session_start_time"] = pd.to_datetime(combined["session_start_time"])

    cutoff = combined.loc[combined["version"] == "Sitara-2", "session_start_time"].min()

    result = run_did(combined, group_col="version", treated_value="Sitara-2", control_value="Sitara-1",
                      date_col="session_start_time", cutoff=cutoff, metric_col="session_duration_minutes")

    summary = pd.Series({k: v for k, v in result.items() if k != "means_table"})
    summary["note"] = (
        "did_estimate is NaN because Sitara-2 has no real pre-cutoff period "
        "(the versions launched sequentially, not concurrently) -- there is "
        "nothing to take a true difference-in-differences of here. "
        "naive_post_only_diff is shown purely to prove the scaffold runs "
        "end-to-end; it is a raw post-period mean difference, NOT a causal "
        "DiD estimate, and should not be interpreted as a version effect."
    )
    summary.to_csv(OUT_DIR / "did_illustrative_sitara2_vs_sitara1.csv",
                    header=["value"])
    result["means_table"].to_csv(OUT_DIR / "did_means_table_sitara2_vs_sitara1.csv")

    plot_parallel_trends(combined, "version", "Sitara-2", "Sitara-1",
                          "session_start_time", cutoff, "session_duration_minutes",
                          "ILLUSTRATIVE ONLY (not concurrent, not causal): "
                          "session duration, Sitara-1 vs Sitara-2 over time",
                          OUT_DIR / "parallel_trends_illustrative_sitara.png")

    print("NOTE: the Sitara-1 vs Sitara-2 DiD run is illustrative only -- the "
          "versions were not run concurrently, so this is a scaffold demo, "
          "not a causal estimate. See module docstring.")
    print(f"Stage 4.7 DiD scaffold written to {OUT_DIR}")
    return result


if __name__ == "__main__":
    run()
