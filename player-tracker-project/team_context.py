"""Team-aware scouting: judge players against a team's formation, playstyle and squad.

A good scout does not ask "who is the best winger?" but "who is the best winger for how WE
play, and would he improve what we already have?". This module turns a team's profile (from
real match data, or described by the user) into scouting weights, finds the positions where
the squad is weakest, and measures whether a candidate beats the current starter.
"""
import numpy as np
import pandas as pd

import scouting as sc

STYLES = ["High press", "Possession build-up", "Counter-attack"]

# Starting slots per position group for common formations (wing-backs count as full-backs)
FORMATION_SLOTS = {
    "4-3-3": {"Centre-back": 2, "Full-back": 2, "Defensive midfielder": 1, "Central midfielder": 2, "Winger": 2, "Forward": 1},
    "4-2-3-1": {"Centre-back": 2, "Full-back": 2, "Defensive midfielder": 2, "Attacking midfielder": 1, "Winger": 2, "Forward": 1},
    "4-4-2": {"Centre-back": 2, "Full-back": 2, "Central midfielder": 2, "Winger": 2, "Forward": 2},
    "4-1-4-1": {"Centre-back": 2, "Full-back": 2, "Defensive midfielder": 1, "Central midfielder": 2, "Winger": 2, "Forward": 1},
    "4-3-2-1": {"Centre-back": 2, "Full-back": 2, "Defensive midfielder": 1, "Central midfielder": 2, "Attacking midfielder": 2, "Forward": 1},
    "4-4-1-1": {"Centre-back": 2, "Full-back": 2, "Central midfielder": 2, "Winger": 2, "Attacking midfielder": 1, "Forward": 1},
    "4-2-2-2": {"Centre-back": 2, "Full-back": 2, "Defensive midfielder": 2, "Attacking midfielder": 2, "Forward": 2},
    "3-5-2": {"Centre-back": 3, "Full-back": 2, "Defensive midfielder": 1, "Central midfielder": 2, "Forward": 2},
    "3-4-3": {"Centre-back": 3, "Full-back": 2, "Central midfielder": 2, "Winger": 2, "Forward": 1},
    "3-4-2-1": {"Centre-back": 3, "Full-back": 2, "Central midfielder": 2, "Attacking midfielder": 2, "Forward": 1},
    "3-4-1-2": {"Centre-back": 3, "Full-back": 2, "Central midfielder": 2, "Attacking midfielder": 1, "Forward": 2},
    "3-1-4-2": {"Centre-back": 3, "Defensive midfielder": 1, "Full-back": 2, "Central midfielder": 2, "Forward": 2},
    "5-3-2": {"Centre-back": 3, "Full-back": 2, "Defensive midfielder": 1, "Central midfielder": 2, "Forward": 2},
    "5-4-1": {"Centre-back": 3, "Full-back": 2, "Central midfielder": 2, "Winger": 2, "Forward": 1},
}
FORMATIONS = list(FORMATION_SLOTS)
NEED_THRESHOLDS = (45,75   # starter percentile below 50 = priority, 80 or more = strong
MIN_TEAMS_FOR_PERCENTILES = 6


# =====================================================
# FORMATION
# =====================================================
def _digits(formation):
    return [int(c) for c in str(formation) if c.isdigit()]


def slots_for_formation(formation):
    """Starting slots per position group. Unknown formations get a rough estimate."""
    if formation in FORMATION_SLOTS:
        return dict(FORMATION_SLOTS[formation])
    d = _digits(formation)
    if len(d) < 3:
        return {}
    back, fwd, mids = d[0], d[-1], sum(d[1:-1])
    slots = {"Centre-back": 3 if back >= 3 else 2, "Full-back": 2, "Forward": fwd}
    slots["Central midfielder"] = max(mids - 2, 1) if back >= 3 else max(mids - 2, 0) + 1
    slots["Winger"] = 2 if back == 4 and mids >= 4 else 0
    return slots


def formation_adjustments(formation):
    """How a formation changes what each position is asked to do, as extra metric weights."""
    d = _digits(formation)
    adj = {}
    if not d:
        return adj
    if d[0] in (3, 5):          # back three or five: full-backs are wing-backs, centre-backs build play
        adj["Full-back"] = {"crosses_p90": 0.3, "carry_dist_p90": 0.3, "prog_carries_p90": 0.2,
                            "dribbles_p90": 0.1, "tackles_p90": -0.1}
        adj["Centre-back"] = {"pass_pct": 0.1, "prog_passes_p90": 0.2, "long_passes_p90": 0.1}
    if len(d) >= 3 and d[1] == 1:   # a lone holding midfielder has to cover a lot of ground
        adj["Defensive midfielder"] = {"interceptions_p90": 0.2, "recoveries_p90": 0.2}
    if d[-1] == 1 and len(d) >= 3:  # a lone striker has to link play and press from the front
        adj["Forward"] = {"passes_p90": 0.1, "pressures_p90": 0.1}
    return adj


# =====================================================
# PLAYSTYLE
# =====================================================
def _percentile(series, value, higher_is_better=True):
    """Where value sits among series, 0-100 (50 when there are too few teams to compare)."""
    series = series.dropna()
    if len(series) < MIN_TEAMS_FOR_PERCENTILES or pd.isna(value):
        return 50.0
    pct = ((series < value).mean() + 0.5 * (series == value).mean()) * 100
    return float(pct if higher_is_better else 100 - pct)


def style_scores_from_team(team_row, teams):
    """How strongly a team plays each style, 0-100, relative to the other teams in the data."""
    p = lambda col, higher=True: _percentile(teams[col], team_row[col], higher)
    return {
        "High press": 0.6 * p("ppda", False) + 0.4 * p("high_press_share"),
        "Possession build-up": 0.7 * p("possession_pct") + 0.3 * p("long_pass_share", False),
        "Counter-attack": 0.4 * p("counter_share") + 0.3 * p("long_pass_share") + 0.3 * p("possession_pct", False),
    }


def style_scores_custom(press, possession, directness):
    """Style scores for a team described by the user (three 0-100 sliders)."""
    return {"High press": float(press), "Possession build-up": float(possession),
            "Counter-attack": float(directness)}


def style_label(scores):
    """A short description such as 'High press + Possession build-up', or 'Balanced'."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [name for name, value in ranked if value >= 60][:2]
    return " + ".join(top) if top else "Balanced"


# =====================================================
# WEIGHTS FOR A TEAM
# =====================================================
def team_weights(group, scores, formation, part="both"):
    """Scouting weights for a position, from the team's formation and playstyle.

    part: "role" (what the position does in this formation), "style" (what the team's way of
    playing asks for) or "both" (the two combined, which is what the fit score uses).
    """
    role = dict(sc.ROLE_WEIGHTS.get(group, {}))
    for metric, delta in formation_adjustments(formation).get(group, {}).items():
        role[metric] = role.get(metric, 0.0) + delta

    # Blend the three styles by how strongly the team plays each; squaring favours the dominant one
    raw = np.array([(max(scores.get(s, 0.0), 0.0) / 100) ** 2 for s in STYLES])
    shares = raw / raw.sum() if raw.sum() > 0 else np.full(len(STYLES), 1 / len(STYLES))
    style = {}
    for name, share in zip(STYLES, shares):
        for metric, w in sc.STYLE_WEIGHTS[name].items():
            style[metric] = style.get(metric, 0.0) + share * w

    chosen = {"role": role, "style": style}.get(part)
    if chosen is None:
        chosen = dict(role)
        for metric, w in style.items():
            chosen[metric] = chosen.get(metric, 0.0) + w
    return {m: float(np.clip(w, -1.0, 1.0)) for m, w in chosen.items() if abs(w) > 1e-9}


# =====================================================
# SQUAD NEEDS
# =====================================================
def _verdict(starter_pct):
    if starter_pct is None:
        return "No qualifying player"
    if starter_pct < NEED_THRESHOLDS[0]:
        return "Priority"
    if starter_pct < NEED_THRESHOLDS[1]:
        return "Solid"
    return "Strong"


def squad_overview(profiles, team, scores, formation, slots, min_minutes=90):
    """For each position the team uses: who starts, how good he is for THIS system, and the need.

    The starter is the team's player with the most minutes in that position. His standing is
    his fit score's percentile among every player in the data at that position.
    """
    rows = []
    for group in sc.POSITION_GROUPS:
        in_use = slots.get(group, 0) >= 0.5
        weights = team_weights(group, scores, formation, "both")
        table, _, _ = sc.fit_scores(profiles, group, weights, min_minutes, small_sample_adjust=True)
        starter, standing, rank = None, None, None
        if not table.empty:
            own = table[table["team"] == team]
            if not own.empty:
                starter = own.sort_values("minutes", ascending=False).iloc[0]
                below = (table["fit_score"] < starter["fit_score"]).mean()
                equal = (table["fit_score"] == starter["fit_score"]).mean()
                standing = float((below + 0.5 * equal) * 100)
                rank = f"{int((table['fit_score'] > starter['fit_score']).sum()) + 1} of {len(table)}"
        rows.append({
            "group": group, "in_use": in_use, "slots": slots.get(group, 0),
            "starter": None if starter is None else starter["player"],
            "starter_minutes": None if starter is None else int(starter["minutes"]),
            "starter_score": None if starter is None else float(starter["fit_score"]),
            "rank": rank, "standing": standing,
            "need": (100.0 - standing) if standing is not None else (100.0 if in_use else 0.0),
            "verdict": _verdict(standing) if in_use else "Not used in this formation",
            "pool": len(table),
        })
    return pd.DataFrame(rows)


# =====================================================
# CANDIDATES FOR A TEAM
# =====================================================
def candidate_ranking(profiles, group, scores, formation, team=None, min_minutes=180,
                      include_versatile=False, small_sample_adjust=True):
    """Rank candidates for a position against a team's system.

    Returns a dict with the ranked table (fit, role and style scores, and the change compared
    with the team's current starter), the per-metric percentiles and raw values, the weights
    used and the current starter row, or None if nobody qualifies.
    """
    weights = team_weights(group, scores, formation, "both")
    table, pct, raw = sc.fit_scores(profiles, group, weights, min_minutes, include_versatile, small_sample_adjust)
    if table.empty:
        return None

    for part, column in (("role", "role_score"), ("style", "style_score")):
        w = team_weights(group, scores, formation, part)
        part_table, _, _ = sc.fit_scores(profiles, group, w, min_minutes, include_versatile, small_sample_adjust)
        lookup = dict(zip(part_table["player_id"], part_table["fit_score"])) if not part_table.empty else {}
        table[column] = table["player_id"].map(lookup)

    starter = None
    if team is not None:
        own = table[table["team"] == team]
        if not own.empty:
            starter = own.sort_values("minutes", ascending=False).iloc[0]
        keep = (table["team"] != team).to_numpy()
        table, pct, raw = (x[keep].reset_index(drop=True) for x in (table, pct, raw))
        if table.empty:
            return None

    if starter is not None:
        table["vs_starter"] = (table["fit_score"] - starter["fit_score"]).round(1)
        table["verdict"] = np.select(
            [table["vs_starter"] >= 10, table["vs_starter"] >= 0],
            ["Clear upgrade", "Marginal upgrade"], "Not an upgrade")
    else:
        table["vs_starter"] = np.nan
        table["verdict"] = "No current starter to compare with"
    return {"table": table, "pct": pct, "raw": raw, "weights": weights, "starter": starter}
