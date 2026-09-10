"""Stage 4.6: PARADISE-style regression.

Regresses available outcome signals (returned_next_session, session_duration,
message_count, gap_until_next_session) against tag rates + trajectory
features. Logistic regression for the boolean outcome, linear regression for
the continuous ones; a decision tree is fit alongside each for an
interpretable, non-linear cross-check. Run per source group and combined.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from astro_analysis import config
from astro_analysis.analysis import groups as groups_mod
from astro_analysis.analysis.plotting import CATEGORICAL, new_fig, save
from astro_analysis.analysis.sentiment import build_trajectories
from astro_analysis.analysis.survival import COVARIATE_TAGS, _session_tag_rates

OUT_DIR = config.RESULTS_DIR / "paradise"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_COLS = ["returned_next_session", "session_duration_minutes",
                "message_count_total", "gap_until_next_session_hours"]


def build_feature_table(messages: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    trajectories = build_trajectories(messages)
    sessions_sorted = sessions.sort_values(["file", "session_number"])
    next_gap = sessions_sorted.groupby("file")["gap_since_previous_session_hours"].shift(-1)
    sessions_sorted = sessions_sorted.assign(gap_until_next_session_hours=next_gap)

    rows = []
    for _, sess in sessions_sorted.iterrows():
        sid = sess["session_id"]
        session_msgs = messages[messages["session_id"] == sid]
        if session_msgs.empty:
            continue
        feats = _session_tag_rates(session_msgs)
        traj = trajectories.get(sid)
        feats["mean_sentiment"] = float(np.mean(traj)) if traj is not None else 0.0
        feats["sentiment_trend"] = float(traj[-1] - traj[0]) if traj is not None and len(traj) > 1 else 0.0

        feats["session_id"] = sid
        feats["session_duration_minutes"] = sess.get("session_duration_minutes")
        feats["message_count_total"] = sess.get("message_count_user", 0) + sess.get("message_count_astrologer", 0)
        feats["gap_until_next_session_hours"] = sess.get("gap_until_next_session_hours")
        feats["returned_next_session"] = int(sess.get("gap_until_next_session_hours") is not None
                                              and not pd.isna(sess.get("gap_until_next_session_hours")))
        rows.append(feats)
    return pd.DataFrame(rows)


FEATURE_COLS = COVARIATE_TAGS + ["mean_sentiment", "sentiment_trend"]


def fit_models(table: pd.DataFrame, outcome: str):
    df = table.dropna(subset=[outcome] + FEATURE_COLS)
    if len(df) < 15:
        return None
    X = df[FEATURE_COLS].to_numpy()
    y = df[outcome].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if outcome == "returned_next_session":
        if len(np.unique(y)) < 2:
            return None
        lin = LogisticRegression(max_iter=2000, C=0.5, penalty="l2")
        lin.fit(X_scaled, y)
        coefs = lin.coef_.ravel()
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, len(df) // 20))
    else:
        lin = LinearRegression()
        lin.fit(X_scaled, y)
        coefs = lin.coef_.ravel()
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(5, len(df) // 20))
    tree.fit(X, y)

    importance = pd.Series(coefs, index=FEATURE_COLS, name="coefficient").sort_values(key=abs, ascending=False)
    tree_importance = pd.Series(tree.feature_importances_, index=FEATURE_COLS,
                                 name="tree_importance").sort_values(ascending=False)
    return {"n": len(df), "linear_coefficients": importance, "tree_importance": tree_importance}


def plot_importance(series: pd.Series, title: str, path, xlabel: str):
    fig, ax = new_fig(figsize=(8, 0.4 * len(series) + 2))
    y = np.arange(len(series))
    colors = [CATEGORICAL[0] if v >= 0 else CATEGORICAL[1] for v in series.values]
    ax.barh(y, series.values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(series.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    save(fig, path)


def run():
    group_frames = groups_mod.get_groups()
    all_tables = {}

    for name, frames in group_frames.items():
        table = build_feature_table(frames["messages"], frames["sessions"])
        all_tables[name] = table
        table.to_csv(OUT_DIR / f"features_{name}.csv", index=False)

        for outcome in OUTCOME_COLS:
            result = fit_models(table, outcome)
            if result is None:
                print(f"[paradise] skipping {name}/{outcome}: insufficient sample")
                continue
            result["linear_coefficients"].to_csv(OUT_DIR / f"coefficients_{name}_{outcome}.csv")
            result["tree_importance"].to_csv(OUT_DIR / f"tree_importance_{name}_{outcome}.csv")
            plot_importance(result["linear_coefficients"],
                             f"{name}: standardized coefficients ({outcome})",
                             OUT_DIR / f"bar_coefficients_{name}_{outcome}.png",
                             "standardized coefficient")

    combined = pd.concat([t.assign(source_group=g) for g, t in all_tables.items()], ignore_index=True)
    combined.to_csv(OUT_DIR / "features_combined.csv", index=False)
    for outcome in OUTCOME_COLS:
        result = fit_models(combined, outcome)
        if result is None:
            continue
        result["linear_coefficients"].to_csv(OUT_DIR / f"coefficients_combined_{outcome}.csv")
        plot_importance(result["linear_coefficients"],
                         f"Combined: standardized coefficients ({outcome})",
                         OUT_DIR / f"bar_coefficients_combined_{outcome}.png",
                         "standardized coefficient")

    print(f"Stage 4.6 PARADISE-style regression written to {OUT_DIR}")


if __name__ == "__main__":
    run()
