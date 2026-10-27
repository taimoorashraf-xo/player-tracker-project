"""Movement indicators for each tracked player, and ranking them against a requirement.

Positions arrive as stabilised pixels (camera motion already removed by `tracking`). Here
they are converted to metres on the pitch, smoothed, and turned into the movement numbers a
scout actually looks at: top speed, distance, high-intensity running, sprints and
accelerations.

Everything is an ESTIMATE from broadcast video:
- Distances rest on an assumed 1.8 m player height and a flat pitch. Expect single-digit
  percentage error at best, worse in close-ups and hand-held shots.
- "Pressing" is a proxy for closing an opponent down. The ball is not tracked, so it cannot
  know who was in possession.
- Passing, dribble success and pitch position cannot be measured from highlights at all.
"""
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

MAX_GAP_S = 0.45            # observations further apart than this are not joined up
SPEED_CAP_MPS = 11.5        # faster than 41 km/h is a tracking glitch, not a player
HI_SPEED_MPS = 5.5          # high-intensity running, about 20 km/h
SPRINT_MPS = 7.0            # sprinting, about 25 km/h
SPRINT_MIN_S = 0.7          # a burst must last this long to count as a sprint
ACCEL_CAP = 8.0             # m/s^2; above this is noise
HIGH_ACCEL = 2.5            # m/s^2, a decisive burst
MIN_OBS_SECONDS = 2.0       # below this there is not enough movement to judge anything
RATE_MIN_SECONDS = 8.0      # sprints and accelerations are too rare to scale up from less
ACCEL_MIN_S = 0.3           # an acceleration must be held this long to count as one burst
PRESS_RADIUS_M = 6.0        # "closing down" counts within this distance of an opponent
CLOSING_MPS = 1.5           # and only while moving towards them at least this fast
SMOOTH_WINDOW_S = 0.5       # trajectory smoothing window

# key -> (label, tooltip, unit, short note shown beside the bars)
VIDEO_METRICS = {
    "top_speed_kmh": ("Top speed", "Fastest sustained speed reached on screen, taken as the "
                      "95th percentile so one glitchy frame cannot set it", "km/h",
                      "95th percentile"),
    "avg_speed_kmh": ("Average speed", "Average speed over the time the player was measured",
                      "km/h", "over measured time"),
    "hi_share": ("High-intensity running", "Share of measured time spent above about 20 km/h",
                 "%", "time above 20 km/h"),
    "distance_m_per_min": ("Distance per minute", "Ground covered per minute on screen, "
                           "a work-rate proxy", "m", "work-rate proxy"),
    "sprints_per_min": ("Sprints per minute", "Separate bursts above about 25 km/h held for at "
                        "least 0.7 s", "", "bursts over 25 km/h"),
    "accels_per_min": ("Accelerations per minute", "Separate bursts of acceleration above "
                       "2.5 m/s^2, a proxy for sharpness", "", "bursts over 2.5 m/s2"),
    "press_share": ("Pressing activity", "Share of measured time spent moving towards a nearby "
                    "opponent; the ball is not tracked", "%", "closing an opponent down"),
}
VIDEO_LABELS = {k: v[0] for k, v in VIDEO_METRICS.items()}
VIDEO_HELP = {k: v[1] for k, v in VIDEO_METRICS.items()}
VIDEO_UNITS = {k: v[2] for k, v in VIDEO_METRICS.items()}
VIDEO_NOTES = {k: v[3] for k, v in VIDEO_METRICS.items()}

VIDEO_STYLES = {
    "Pace and penetration": {"top_speed_kmh": 1.0, "sprints_per_min": 0.8, "accels_per_min": 0.6,
                             "hi_share": 0.5},
    "High press": {"press_share": 1.0, "distance_m_per_min": 0.7, "hi_share": 0.6,
                   "accels_per_min": 0.5},
    "Engine (box to box)": {"distance_m_per_min": 1.0, "hi_share": 0.7, "avg_speed_kmh": 0.6,
                            "sprints_per_min": 0.4},
    "Explosive (short bursts)": {"accels_per_min": 1.0, "sprints_per_min": 0.8, "top_speed_kmh": 0.5},
    "All-round athlete": {k: 0.6 for k in VIDEO_METRICS},
}


# =====================================================
# ONE PLAYER'S TRAJECTORY
# =====================================================
def ground_track(obs, scale, centre_x):
    """Turn one track's stabilised pixel observations into a smoothed path in metres.

    `obs` is a list of (time, foot x, foot y, box height, touches frame edge).
    Returns a DataFrame with time, position (x, y in metres), speed and acceleration, split
    into continuous runs so that a gap in the tracking never looks like a teleport.
    """
    columns = ["t", "x", "y", "speed", "accel", "run"]
    if len(obs) < 3:
        return pd.DataFrame(columns=columns)

    data = pd.DataFrame(obs, columns=["t", "px", "py", "h", "edge"]).sort_values("t")
    data = data[~data["edge"].astype(bool)].reset_index(drop=True)   # clipped boxes give bad feet
    if len(data) < 3:
        return pd.DataFrame(columns=columns)

    x, y = scale.ground_xy(data["px"].to_numpy(), data["py"].to_numpy(), centre_x)
    data["x"], data["y"] = x, y

    # Split the track wherever it cannot be one continuous movement: a break in time, or a
    # step too big for a human. The second case is usually the tracker swapping two players
    # who crossed. Clipping such a jump to the maximum speed instead would quietly record
    # it as the player's best sprint, which is how a tracking error becomes a scouting one.
    gap = data["t"].diff()
    leap = np.hypot(data["x"].diff(), data["y"].diff()) / gap
    data["run"] = ((gap > MAX_GAP_S) | (leap > SPEED_CAP_MPS)).fillna(False).cumsum()

    pieces = []
    for _, run in data.groupby("run"):
        if len(run) < 3:
            continue
        run = run.copy()
        dt = float(np.median(run["t"].diff().dropna())) or 0.1
        window = int(round(SMOOTH_WINDOW_S / dt))
        window = max(3, window + (window + 1) % 2)          # odd, at least 3
        if window <= len(run) - 1:
            poly = 2 if window > 3 else 1
            run["x"] = savgol_filter(run["x"], window, poly)
            run["y"] = savgol_filter(run["y"], window, poly)
        step = np.hypot(run["x"].diff(), run["y"].diff())
        run["speed"] = step / run["t"].diff()
        run["accel"] = (run["speed"].diff() / run["t"].diff()).clip(-ACCEL_CAP, ACCEL_CAP)
        pieces.append(run.dropna(subset=["speed"]))

    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True)[columns]


def _bursts(times, values, threshold, min_seconds):
    """Count separate stretches where a value stays above a threshold for long enough."""
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)
    above = values >= threshold
    count, start = 0, None
    for i, hot in enumerate(above):
        if hot and start is None:
            start = times[i]
        elif not hot and start is not None:
            if times[i - 1] - start >= min_seconds:
                count += 1
            start = None
    if start is not None and times[-1] - start >= min_seconds:
        count += 1
    return count


def movement_metrics(track):
    """Speed, distance, intensity, sprints and accelerations for one player's path."""
    if track.empty:
        return None
    dt = track["t"].diff().fillna(np.median(track["t"].diff().dropna()) if len(track) > 1 else 0.1)
    dt = dt.clip(upper=MAX_GAP_S)
    observed = float(dt.sum())
    if observed < MIN_OBS_SECONDS:
        return None

    speed = track["speed"].to_numpy()
    distance = float((speed * dt).sum())
    per_min = 60.0 / observed
    rates_ok = observed >= RATE_MIN_SECONDS
    return {
        "observed_s": observed,
        "top_speed_kmh": float(np.percentile(speed, 95) * 3.6),
        "avg_speed_kmh": float(np.average(speed, weights=dt) * 3.6),
        "hi_share": float(dt[speed >= HI_SPEED_MPS].sum() / observed * 100),
        "distance_m": distance,
        "distance_m_per_min": distance * per_min,
        # Both are counts of separate bursts. Counting every frame above the threshold
        # instead would turn one three-second surge into thirty "accelerations". They are
        # left blank for short spells, where one burst would scale up to a wild per-minute
        # figure rather than anything you could compare between players.
        "sprints_per_min": _bursts(track["t"].to_numpy(), speed,
                                   SPRINT_MPS, SPRINT_MIN_S) * per_min if rates_ok else np.nan,
        "accels_per_min": _bursts(track["t"].to_numpy(), track["accel"].to_numpy(),
                                  HIGH_ACCEL, ACCEL_MIN_S) * per_min if rates_ok else np.nan,
    }


# =====================================================
# PRESSING: CLOSING DOWN AN OPPONENT
# =====================================================
def press_metrics(records, team_of, scale, centre_x):
    """Share of each player's time spent moving towards a nearby opponent.

    `records` is one entry per analysed frame: {"t", "players": {key: {...}}}. Only players
    with a known team are compared, and only against the other team.
    """
    previous, observed, pressing = {}, defaultdict(float), defaultdict(float)

    for record in records:
        now = record["t"]
        present = {k: p for k, p in record["players"].items()
                   if team_of.get(k) and not p["edge"]}
        if not present:
            continue
        keys = list(present)
        px = np.array([present[k]["pos"][0] for k in keys], dtype=float)
        py = np.array([present[k]["pos"][1] for k in keys], dtype=float)
        gx, gy = scale.ground_xy(px, py, centre_x)
        ground = np.column_stack([gx, gy])
        teams = np.array([team_of[k] for k in keys])

        for i, key in enumerate(keys):
            last = previous.get(key)
            previous[key] = (now, ground[i])
            if last is None or not 0 < now - last[0] <= MAX_GAP_S:
                continue
            gap = now - last[0]
            observed[key] += gap
            velocity = (ground[i] - last[1]) / gap
            rivals = np.flatnonzero(teams != teams[i])
            if not len(rivals):
                continue
            offsets = ground[rivals] - ground[i]
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
            nearest = int(np.argmin(distances))
            if 0 < distances[nearest] <= PRESS_RADIUS_M:
                closing = float(velocity @ (offsets[nearest] / distances[nearest]))
                if closing >= CLOSING_MPS:
                    pressing[key] += gap

    return {k: {"press_share": pressing[k] / observed[k] * 100}
            for k in observed if observed[k] > 0}


# =====================================================
# TEAMS FROM KIT COLOUR
# =====================================================
def _lab_to_hex(lab):
    b, g, r = cv2.cvtColor(np.uint8([[np.clip(lab, 0, 255)]]), cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def cluster_teams(colors):
    """Group tracks into Team A, Team B and Other by average shirt colour.

    `colors` maps a track key to (mean Lab colour, number of frames it was measured over).
    Returns (key -> team name, team name -> hex swatch).
    """
    keys = [k for k, (_, n) in colors.items() if n >= 3]
    if not keys:
        return {}, {}

    data = np.float32([colors[k][0] for k in keys])
    frames = np.array([colors[k][1] for k in keys], dtype=float)
    clusters = 3 if len(keys) >= 6 else (2 if len(keys) >= 3 else 1)

    if clusters == 1:
        labels, centres = np.zeros(len(keys), dtype=int), data.mean(axis=0, keepdims=True)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, raw, centres = cv2.kmeans(data, clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = raw.ravel()

    # The two biggest groups are the teams; anything left over is usually officials
    order = sorted(range(len(centres)), key=lambda c: -frames[labels == c].sum())
    names = ["Team A", "Team B", "Other"]
    name_of = {c: names[rank] for rank, c in enumerate(order)}
    team_of = {key: name_of[int(labels[i])] for i, key in enumerate(keys)}
    return team_of, {name_of[c]: _lab_to_hex(centres[c]) for c in order}


# =====================================================
# RANKING
# =====================================================
def rank_candidates(df, weights):
    """Score candidates against the weights. Returns (ranked table, percentiles).

    Each measurement becomes a percentile among the players in this video, and the fit
    score is the weighted average of those percentiles.
    """
    weights = {m: w for m, w in weights.items() if w != 0 and m in df.columns}
    if df.empty or not weights:
        return pd.DataFrame(), pd.DataFrame()
    pool = df.reset_index(drop=True)

    pct = pd.DataFrame(index=pool.index)
    for metric, weight in weights.items():
        ranked = pool[metric].rank(pct=True) * 100
        pct[metric] = ranked if weight > 0 else 100 - ranked
    pct = pct.fillna(50.0)

    total = sum(abs(w) for w in weights.values())
    score = sum(pct[m] * abs(w) for m, w in weights.items()) / total
    table = pool.copy()
    table["fit_score"] = score.round(1)
    order = table["fit_score"].sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True), pct.loc[order].reset_index(drop=True)


def strengths_gaps(pct_row, weights, labels=None, min_gap_weight=0.5):
    """What a player is good at and weak in, as lists of (label, percentile)."""
    labels = labels or VIDEO_LABELS
    important = {m: w for m, w in weights.items() if w != 0 and m in pct_row.index}
    by_value = sorted(important, key=lambda m: abs(important[m]) * pct_row[m], reverse=True)
    strengths = [(labels[m], pct_row[m]) for m in by_value if pct_row[m] >= 70][:3]
    by_need = sorted(important, key=lambda m: abs(important[m]), reverse=True)
    gaps = [(labels[m], pct_row[m]) for m in by_need
            if pct_row[m] <= 35 and abs(important[m]) >= min_gap_weight][:2]
    return strengths, gaps


def explain_video(row, pct_row, weights):
    """One or two sentences on why a player scored what they did."""
    strengths, gaps = strengths_gaps(pct_row, weights)
    parts = []
    if strengths:
        parts.append("Stands out for " + ", ".join(
            f"{label.lower()} (percentile {int(round(p))})" for label, p in strengths) + ".")
    else:
        parts.append("Nothing stands out against these requirements.")
    if gaps:
        parts.append("Weaker on " + ", ".join(
            f"{label.lower()} (percentile {int(round(p))})" for label, p in gaps) + ".")
    if row["observed_s"] < 5:
        parts.append(f"Only {row['observed_s']:.1f} s of usable footage, so treat it as a hint.")
    return " ".join(parts)
