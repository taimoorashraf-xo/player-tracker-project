"""Scouting analysis built on free StatsBomb open event data.

Event data records every pass, carry, pressure and so on, so it gives real passing and
off-the-ball activity numbers. It does NOT contain distance run or sprint speed (that
needs tracking data), so work rate and pace are expressed through event-based proxies.
"""
import os

import numpy as np
import pandas as pd

# =====================================================
# METRICS
# =====================================================
# key -> (label, description, count column it is built from)
RATE_METRICS = {
    "passes_p90": ("Passes", "Passes attempted per 90", "passes"),
    "prog_passes_p90": ("Progressive passes", "Completed passes that move the ball 10+ yards towards goal, per 90", "prog_passes"),
    "key_passes_p90": ("Key passes", "Passes that lead to a shot, per 90", "key_passes"),
    "final_third_passes_p90": ("Passes into final third", "Completed passes from outside into the final third, per 90", "final_third_passes"),
    "long_passes_p90": ("Long passes", "Passes of 30+ yards attempted, per 90", "long_passes"),
    "crosses_p90": ("Crosses", "Crosses attempted per 90", "crosses"),
    "pressures_p90": ("Pressures", "Times the player pressed an opponent, per 90 (work-rate proxy)", "pressures"),
    "counterpress_p90": ("Counter-presses", "Pressures right after losing the ball, per 90 (work-rate proxy)", "counterpresses"),
    "recoveries_p90": ("Ball recoveries", "Loose balls won back, per 90", "recoveries"),
    "interceptions_p90": ("Interceptions", "Successful interceptions per 90", "interceptions"),
    "tackles_p90": ("Tackles", "Tackles attempted per 90", "tackles"),
    "carries_p90": ("Carries", "Times the player drove with the ball, per 90", "carries"),
    "prog_carries_p90": ("Progressive carries", "Carries that move the ball 10+ yards towards goal, per 90 (pace proxy)", "prog_carries"),
    "carry_dist_p90": ("Carry distance", "Yards covered while carrying the ball, per 90 (pace proxy)", "carry_dist"),
    "dribbles_p90": ("Successful dribbles", "Opponents beaten on the dribble, per 90 (pace proxy)", "dribbles"),
    "shots_p90": ("Shots", "Shots per 90", "shots"),
    "xg_p90": ("Expected goals (xG)", "Quality of chances taken, per 90", "xg"),
    "goals_p90": ("Goals", "Goals per 90", "goals"),
    "turnovers_p90": ("Ball losses", "Miscontrols and times dispossessed, per 90 (lower is better)", "turnovers"),
}
OTHER_METRICS = {
    "pass_pct": ("Pass completion %", "Share of passes completed"),
    "avg_x": ("Average position (0 = own goal, 120 = opponent goal)", "How high up the pitch the player's actions are"),
}
METRIC_LABELS = {k: v[0] for k, v in RATE_METRICS.items()}
METRIC_LABELS.update({k: v[0] for k, v in OTHER_METRICS.items()})
METRIC_HELP = {k: v[1] for k, v in RATE_METRICS.items()}
METRIC_HELP.update({k: v[1] for k, v in OTHER_METRICS.items()})
SUM_COLS = sorted({v[2] for v in RATE_METRICS.values()} | {"pass_ok", "sum_x", "n_loc"})

# =====================================================
# POSITIONS AND SYSTEMS
# =====================================================
POSITION_GROUPS = [
    "Centre-back", "Full-back", "Defensive midfielder", "Central midfielder",
    "Attacking midfielder", "Winger", "Forward",
]

# What each position is asked to do (weights from -1 to 1; negative = lower is better)
ROLE_WEIGHTS = {
    "Centre-back": {"interceptions_p90": 0.8, "tackles_p90": 0.5, "pass_pct": 0.6, "long_passes_p90": 0.5,
                    "prog_passes_p90": 0.6, "recoveries_p90": 0.4, "turnovers_p90": -0.5},
    "Full-back": {"prog_passes_p90": 0.6, "crosses_p90": 0.6, "prog_carries_p90": 0.6, "carry_dist_p90": 0.4,
                  "key_passes_p90": 0.4, "tackles_p90": 0.4, "recoveries_p90": 0.4, "dribbles_p90": 0.3},
    "Defensive midfielder": {"recoveries_p90": 0.8, "interceptions_p90": 0.7, "tackles_p90": 0.6, "pass_pct": 0.6,
                             "prog_passes_p90": 0.6, "passes_p90": 0.4, "turnovers_p90": -0.5},
    "Central midfielder": {"passes_p90": 0.5, "pass_pct": 0.4, "prog_passes_p90": 0.7, "key_passes_p90": 0.4,
                           "carries_p90": 0.4, "prog_carries_p90": 0.5, "recoveries_p90": 0.5, "pressures_p90": 0.3},
    "Attacking midfielder": {"key_passes_p90": 0.8, "xg_p90": 0.5, "shots_p90": 0.4, "dribbles_p90": 0.5,
                             "final_third_passes_p90": 0.6, "prog_carries_p90": 0.4},
    "Winger": {"dribbles_p90": 0.8, "prog_carries_p90": 0.7, "carry_dist_p90": 0.5, "key_passes_p90": 0.6,
               "xg_p90": 0.6, "shots_p90": 0.4, "crosses_p90": 0.4},
    "Forward": {"xg_p90": 0.9, "shots_p90": 0.7, "goals_p90": 0.5, "key_passes_p90": 0.3,
                "dribbles_p90": 0.3, "pressures_p90": 0.3},
}

# What each way of playing adds on top of the position's job
STYLE_WEIGHTS = {
    "Balanced": {},
    "High press": {"pressures_p90": 0.9, "counterpress_p90": 0.8, "recoveries_p90": 0.6, "tackles_p90": 0.3},
    "Possession build-up": {"passes_p90": 0.6, "pass_pct": 0.7, "prog_passes_p90": 0.5, "turnovers_p90": -0.5},
    "Counter-attack": {"prog_carries_p90": 0.7, "carry_dist_p90": 0.6, "dribbles_p90": 0.5,
                       "prog_passes_p90": 0.4, "long_passes_p90": 0.4},
}


def position_group(position):
    """Map a StatsBomb position name to one of our position groups."""
    if not isinstance(position, str):
        return "Other"
    if position == "Goalkeeper":
        return "Goalkeeper"
    if "Center Back" in position:
        return "Centre-back"
    if "Back" in position:                      # Left/Right Back, Left/Right Wing Back
        return "Full-back"
    if "Defensive Midfield" in position:
        return "Defensive midfielder"
    if "Attacking Midfield" in position:
        return "Attacking midfielder"
    if "Center Midfield" in position:
        return "Central midfielder"
    if "Wing" in position or position in ("Left Midfield", "Right Midfield"):
        return "Winger"
    if "Forward" in position or "Striker" in position:
        return "Forward"
    return "Other"


def default_weights(group, style):
    """Position job plus style requirements, combined and kept within -1..1."""
    weights = dict(ROLE_WEIGHTS.get(group, {}))
    for metric, w in STYLE_WEIGHTS.get(style, {}).items():
        weights[metric] = weights.get(metric, 0.0) + w
    return {m: float(np.clip(w, -1.0, 1.0)) for m, w in weights.items()}


# =====================================================
# EVENTS -> PER-MATCH PLAYER STATS
# =====================================================
def _col(ev, name):
    return ev[name] if name in ev.columns else pd.Series(np.nan, index=ev.index)


def _xy(series):
    """Float arrays (x, y) from a column of [x, y] lists, NaN where missing."""
    x = np.full(len(series), np.nan)
    y = np.full(len(series), np.nan)
    for i, v in enumerate(series.to_numpy()):
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            x[i], y[i] = v[0], v[1]
    return x, y


def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def player_minutes(ev):
    """Minutes played per player id, from the starting XI, substitutions and red cards."""
    end = float(ev["minute"].max()) if len(ev) else 90.0
    start, stop = {}, {}

    for _, row in ev[ev["type"] == "Starting XI"].iterrows():
        tactics = row.get("tactics")
        if isinstance(tactics, dict):
            for entry in tactics.get("lineup", []):
                start[int(entry["player"]["id"])] = 0.0

    for _, row in ev[ev["type"] == "Substitution"].iterrows():
        minute = float(row["minute"])
        if pd.notna(row.get("player_id")):
            stop[int(row["player_id"])] = minute
        if pd.notna(row.get("substitution_replacement_id")):
            start[int(row["substitution_replacement_id"])] = minute

    red = {"Red Card", "Second Yellow"}
    for card_col in ("foul_committed_card", "bad_behaviour_card"):
        cards = ev[_col(ev, card_col).isin(red)]
        for _, row in cards.iterrows():
            if pd.notna(row.get("player_id")):
                pid = int(row["player_id"])
                stop[pid] = min(stop.get(pid, end), float(row["minute"]))

    minutes = {}
    for pid in set(start) | set(stop):
        minutes[pid] = max(0.0, stop.get(pid, end) - start.get(pid, 0.0))
    return minutes


def match_player_stats(ev):
    """Raw event counts, minutes and average position for each player in one match."""
    minutes = player_minutes(ev)
    d = ev[ev["player_id"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["player_id"] = d["player_id"].astype(int)
    idx = d.index
    typ = d["type"]

    x, y = _xy(_col(d, "location"))
    ex, _ = _xy(_col(d, "pass_end_location"))
    cx, cy = _xy(_col(d, "carry_end_location"))
    x, y, ex, cx, cy = (pd.Series(a, index=idx) for a in (x, y, ex, cx, cy))

    is_pass = typ.eq("Pass")
    pass_ok = is_pass & _col(d, "pass_outcome").isna()
    is_carry = typ.eq("Carry")
    is_shot = typ.eq("Shot")
    has_loc = x.notna()

    f = pd.DataFrame({"player_id": d["player_id"]}, index=idx)
    f["passes"] = is_pass
    f["pass_ok"] = pass_ok
    f["prog_passes"] = pass_ok & ((ex - x) >= 10)
    f["key_passes"] = is_pass & (_col(d, "pass_shot_assist").eq(True) | _col(d, "pass_goal_assist").eq(True))
    f["final_third_passes"] = pass_ok & (ex >= 80) & (x < 80)
    f["long_passes"] = is_pass & (_col(d, "pass_length") >= 30)
    f["crosses"] = is_pass & _col(d, "pass_cross").eq(True)
    f["pressures"] = typ.eq("Pressure")
    f["counterpresses"] = typ.eq("Pressure") & _col(d, "counterpress").eq(True)
    f["recoveries"] = typ.eq("Ball Recovery") & ~_col(d, "ball_recovery_recovery_failure").eq(True)
    f["interceptions"] = typ.eq("Interception") & _col(d, "interception_outcome").isin(
        ["Won", "Success In Play", "Success Out"])
    f["tackles"] = typ.eq("Duel") & _col(d, "duel_type").eq("Tackle")
    f["carries"] = is_carry
    f["prog_carries"] = is_carry & ((cx - x) >= 10)
    f["carry_dist"] = np.where(is_carry, np.hypot(cx - x, cy - y), 0.0)
    f["dribbles"] = typ.eq("Dribble") & _col(d, "dribble_outcome").eq("Complete")
    f["shots"] = is_shot
    f["xg"] = np.where(is_shot, _col(d, "shot_statsbomb_xg").fillna(0.0), 0.0)
    f["goals"] = is_shot & _col(d, "shot_outcome").eq("Goal")
    f["turnovers"] = typ.isin(["Miscontrol", "Dispossessed"])
    f["sum_x"] = np.where(has_loc, x, 0.0)
    f["n_loc"] = has_loc

    counts = f.groupby("player_id")[SUM_COLS].sum().astype(float)
    meta = d.groupby("player_id").agg(
        player=("player", "first"), team=("team", "first"), position=("position", _mode))

    out = meta.join(counts)
    first_last = d.groupby("player_id")["minute"].agg(["min", "max"])
    fallback = (first_last["max"] - first_last["min"]).clip(lower=1.0)
    out["minutes"] = [minutes.get(pid, fallback[pid]) for pid in out.index]
    return out


# =====================================================
# DATA LOADING (cached on disk, one small file per match)
# =====================================================
def list_competitions():
    """Competition-seasons with free data, as a DataFrame with a 'label' column."""
    from statsbombpy import sb
    comps = sb.competitions()
    comps = comps[comps["competition_gender"] == "male"].copy()
    comps["label"] = comps["competition_name"] + " " + comps["season_name"].astype(str)
    return comps.sort_values("label").reset_index(drop=True)


def fetch_match_stats(match_id, cache_dir):
    """Per-player stats for one match, downloaded once and then read from disk."""
    path = os.path.join(cache_dir, f"{int(match_id)}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    from statsbombpy import sb
    stats = match_player_stats(sb.events(match_id=int(match_id)))
    os.makedirs(cache_dir, exist_ok=True)
    stats.to_pickle(path)
    return stats


def match_ids_for(selections):
    """Match ids for a list of (competition_id, season_id) pairs."""
    from statsbombpy import sb
    ids = []
    for comp_id, season_id in selections:
        matches = sb.matches(competition_id=int(comp_id), season_id=int(season_id))
        ids += sorted(matches["match_id"].astype(int).tolist())
    return ids


def aggregate(frames):
    """Combine per-match stats into one row per player."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames)
    sums = allf.groupby(level=0)[SUM_COLS + ["minutes"]].sum()
    meta = allf.groupby(level=0).agg(
        player=("player", "last"), team=("team", "last"), position=("position", _mode),
        matches=("minutes", "size"))
    return meta.join(sums)


def build_profiles(agg):
    """Per-90 rates and the other metrics, plus each player's position group."""
    p = agg[agg["minutes"] > 0].copy()
    per90 = 90.0 / p["minutes"]
    for key, (_, _, col) in RATE_METRICS.items():
        p[key] = p[col] * per90
    p["pass_pct"] = np.where(p["passes"] > 0, p["pass_ok"] / p["passes"].clip(lower=1) * 100, np.nan)
    p["avg_x"] = np.where(p["n_loc"] > 0, p["sum_x"] / p["n_loc"].clip(lower=1), np.nan)
    p["position_group"] = p["position"].map(position_group)
    keep = ["player", "team", "position", "position_group", "matches", "minutes"] + \
           list(RATE_METRICS) + list(OTHER_METRICS)
    return p[keep].reset_index(drop=True)


def build_dataset(selections, cache_dir, max_matches=None, progress=None):
    """Download (or read from cache) the matches and return (profiles, skipped_matches)."""
    ids = match_ids_for(selections)
    if max_matches:
        ids = ids[:max_matches]
    frames, skipped = [], 0
    for i, mid in enumerate(ids, start=1):
        try:
            frames.append(fetch_match_stats(mid, cache_dir))
        except Exception:
            skipped += 1
        if progress:
            progress(i, len(ids))
    agg = aggregate(frames)
    return (build_profiles(agg) if not agg.empty else pd.DataFrame()), skipped


# =====================================================
# FIT SCORING
# =====================================================
def fit_scores(profiles, group, weights, min_minutes):
    """Score every player in the position group against the requirements.

    Each metric is turned into a percentile among players in the same position group
    (flipped when the weight is negative), and the score is the weighted average, 0-100.
    Returns (ranked table, per-metric adjusted percentiles, per-metric raw values), all in
    the same row order, or three empty frames.
    """
    weights = {m: w for m, w in weights.items() if w != 0 and m in profiles.columns}
    pool = profiles[(profiles["position_group"] == group) & (profiles["minutes"] >= min_minutes)].copy()
    if pool.empty or not weights:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pool = pool.reset_index(drop=True)

    pct = pd.DataFrame(index=pool.index)
    for metric, w in weights.items():
        ranked = pool[metric].rank(pct=True) * 100
        pct[metric] = ranked if w > 0 else 100 - ranked
    pct = pct.fillna(50.0)

    total = sum(abs(w) for w in weights.values())
    score = sum(pct[m] * abs(w) for m, w in weights.items()) / total

    table = pool[["player", "team", "position", "minutes"]].copy()
    table["minutes"] = table["minutes"].round(0).astype(int)
    table["fit_score"] = score.round(1)
    order = table["fit_score"].sort_values(ascending=False).index
    raw = pool[list(weights)]
    return (table.loc[order].reset_index(drop=True), pct.loc[order].reset_index(drop=True),
            raw.loc[order].reset_index(drop=True))


def explain(player_row, pct_row, weights):
    """Plain-language summary of why a player does or doesn't fit the requirements."""
    ordinal = lambda v: f"percentile {int(round(v))}"
    important = {m: w for m, w in weights.items() if w != 0 and m in pct_row.index}
    by_value = sorted(important, key=lambda m: abs(important[m]) * pct_row[m], reverse=True)
    strengths = [m for m in by_value if pct_row[m] >= 70][:3]
    by_need = sorted(important, key=lambda m: abs(important[m]), reverse=True)
    gaps = [m for m in by_need if pct_row[m] <= 35 and abs(important[m]) >= 0.3][:2]

    parts = []
    if strengths:
        parts.append("Strong where it matters: " + ", ".join(
            f"{METRIC_LABELS[m].lower()} ({ordinal(pct_row[m])})" for m in strengths) + ".")
    else:
        parts.append("No standout strengths against these requirements.")
    if gaps:
        parts.append("Weaker in: " + ", ".join(
            f"{METRIC_LABELS[m].lower()} ({ordinal(pct_row[m])})" for m in gaps) + ".")
    if player_row["minutes"] < 450:
        parts.append(f"Small sample ({int(player_row['minutes'])} minutes), so treat this with caution.")
    return " ".join(parts)
