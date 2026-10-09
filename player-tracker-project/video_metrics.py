"""Player indicators estimated from broadcast video, and ranking against requirements.

Everything here is an ESTIMATE from edited video, meant to show what is possible:

- Speed uses each player's apparent height (assumed 1.8 m) as a local scale, after the
  camera pan has been removed, so no pitch calibration is needed. Vertical motion in the
  image is not corrected for camera angle, so depth-wise running is under-estimated.
- "Pressing" is a proxy: time spent close to an opponent while moving towards them. The ball
  is not tracked, so it cannot tell whether the opponent had the ball.
- Passing, dribble success and exact position cannot be measured from highlights.
"""
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

PLAYER_HEIGHT_M = 1.8
MAX_GAP_S = 0.45            # observations further apart than this are not joined into a speed
SPEED_CAP_MPS = 12.0        # faster than this (43 km/h) is treated as a tracking glitch
HI_SPEED_MPS = 5.5          # high-intensity running threshold (about 20 km/h)
MIN_OBS_SECONDS = 1.5       # a player must be measurable for at least this long
CONTACT_HEIGHTS = 3.0       # "close to an opponent" = within 3 player heights (about 5 m)
PRESS_HEIGHTS = 3.0
CLOSING_MPS = 1.5           # moving towards the opponent at least this fast counts as pressing

# metric key -> (label, help)
VIDEO_METRICS = {
    "avg_speed_kmh": ("Average speed (km/h)", "Average running speed while on screen"),
    "top_speed_kmh": ("Top speed (km/h)", "Fast running speed reached on screen (95th percentile)"),
    "hi_share": ("High-intensity running (%)", "Share of on-screen time spent running faster than about 20 km/h"),
    "distance_m_per_min": ("Distance per minute (m)", "Distance covered per minute on screen (work-rate proxy)"),
    "press_share": ("Pressing activity (%)", "Share of on-screen time spent closing in on an opponent (pressing proxy)"),
    "contact_share": ("Time close to opponents (%)", "Share of on-screen time within about 5 m of an opponent"),
}
VIDEO_LABELS = {k: v[0] for k, v in VIDEO_METRICS.items()}
VIDEO_HELP = {k: v[1] for k, v in VIDEO_METRICS.items()}

VIDEO_STYLES = {
    "High press": {"press_share": 1.0, "avg_speed_kmh": 0.6, "distance_m_per_min": 0.6,
                   "hi_share": 0.6, "contact_share": 0.3},
    "Counter-attack (pace)": {"top_speed_kmh": 1.0, "hi_share": 0.7, "avg_speed_kmh": 0.4},
    "Work rate (box-to-box)": {"distance_m_per_min": 1.0, "avg_speed_kmh": 0.7, "hi_share": 0.5},
    "Balanced athletic": {k: 0.5 for k in VIDEO_METRICS},
}


# =====================================================
# SPEED AND DISTANCE FROM ONE PLAYER'S TRACK
# =====================================================
def kinematics(obs):
    """Speed segments for one track.

    obs: list of (t, x, y, h, edge): time in seconds, foot position in camera-stabilised pixels,
    box height in pixels, and whether the box touched the frame edge.
    Returns a DataFrame with t, dt, vx, vy, speed (metres per second) for valid segments.
    """
    cols = ["t", "dt", "vx", "vy", "speed"]
    if len(obs) < 2:
        return pd.DataFrame(columns=cols)

    a = pd.DataFrame(obs, columns=["t", "x", "y", "h", "edge"]).sort_values("t").reset_index(drop=True)
    a["edge"] = a["edge"].astype(bool)
    a["h"] = a["h"].astype(float).where(~a["edge"])
    a["hs"] = a["h"].rolling(5, center=True, min_periods=1).median()

    dt = a["t"].diff()
    scale = PLAYER_HEIGHT_M / ((a["hs"] + a["hs"].shift()) / 2)
    vx = a["x"].diff() / dt * scale
    vy = a["y"].diff() / dt * scale
    speed = np.hypot(vx, vy)

    both_inside = ~a["edge"] & ~a["edge"].shift(fill_value=True)
    valid = (dt > 0) & (dt <= MAX_GAP_S) & both_inside & scale.notna() & (speed <= SPEED_CAP_MPS)

    seg = pd.DataFrame({"t": a["t"], "dt": dt, "vx": vx, "vy": vy, "speed": speed})[valid].reset_index(drop=True)
    if not seg.empty:
        seg["speed"] = seg["speed"].rolling(3, center=True, min_periods=1).median()
    return seg


def track_metrics(obs):
    """Speed, distance and high-intensity running for one track, or None if seen too briefly."""
    seg = kinematics(obs)
    if seg.empty or seg["dt"].sum() < MIN_OBS_SECONDS:
        return None
    observed = float(seg["dt"].sum())
    distance = float((seg["speed"] * seg["dt"]).sum())
    return {
        "observed_s": observed,
        "avg_speed_kmh": float(np.average(seg["speed"], weights=seg["dt"]) * 3.6),
        "top_speed_kmh": float(np.percentile(seg["speed"], 95) * 3.6),
        "distance_m": distance,
        "distance_m_per_min": distance / observed * 60.0,
        "hi_share": float(seg.loc[seg["speed"] >= HI_SPEED_MPS, "dt"].sum() / observed * 100.0),
    }


# =====================================================
# CLOSENESS TO OPPONENTS (PRESSING PROXY)
# =====================================================
def contact_metrics(records, team_of):
    """Time spent close to an opponent, and time spent closing in on one.

    records: per analysed frame {"t", "players": {key: {"pos": (x, y), "h": px, "edge": bool}}}
    team_of: key -> team name; only players with a team in two distinct teams are compared.
    Returns key -> {"contact_share", "press_share"} in percent.
    """
    prev = {}
    observed, contact, press = defaultdict(float), defaultdict(float), defaultdict(float)

    for rec in records:
        t = rec["t"]
        items = {k: p for k, p in rec["players"].items() if team_of.get(k) and not p["edge"]}
        keys = list(items)
        if not keys:
            continue
        pos = np.array([items[k]["pos"] for k in keys], dtype=float)
        teams = np.array([team_of[k] for k in keys])

        for i, k in enumerate(keys):
            last = prev.get(k)
            if last is not None and 0 < t - last[0] <= MAX_GAP_S:
                dt = t - last[0]
                vel = (pos[i] - last[1]) / dt
                observed[k] += dt
                rivals = np.where(teams != teams[i])[0]
                if len(rivals):
                    dists = np.hypot(*(pos[rivals] - pos[i]).T)
                    j = int(np.argmin(dists))
                    h = items[k]["h"]
                    if dists[j] / h <= CONTACT_HEIGHTS:
                        contact[k] += dt
                    if dists[j] / h <= PRESS_HEIGHTS and dists[j] > 0:
                        toward = (pos[rivals[j]] - pos[i]) / dists[j]
                        closing = float(vel @ toward) * (PLAYER_HEIGHT_M / h)
                        if closing >= CLOSING_MPS:
                            press[k] += dt
            prev[k] = (t, pos[i])

    return {
        k: {"contact_share": contact[k] / observed[k] * 100.0, "press_share": press[k] / observed[k] * 100.0}
        for k in observed if observed[k] > 0
    }


# =====================================================
# TEAMS FROM KIT COLOUR
# =====================================================
def _lab_to_hex(lab):
    pixel = np.uint8([[np.clip(lab, 0, 255)]])
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def cluster_teams(colors):
    """Group tracks into Team A, Team B and Other by average shirt colour.

    colors: key -> (mean Lab colour as 3 floats in OpenCV's uint8 Lab scale, number of frames).
    Returns (key -> team name, team name -> hex swatch).
    """
    keys = [k for k, (_, n) in colors.items() if n >= 3]
    if not keys:
        return {}, {}

    data = np.float32([colors[k][0] for k in keys])
    frames = np.array([colors[k][1] for k in keys], dtype=float)
    k_clusters = 3 if len(keys) >= 6 else (2 if len(keys) >= 3 else 1)

    if k_clusters == 1:
        labels, centers = np.zeros(len(keys), dtype=int), data.mean(axis=0, keepdims=True)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, lab, centers = cv2.kmeans(data, k_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = lab.ravel()

    order = sorted(range(len(centers)), key=lambda c: -frames[labels == c].sum())
    names = ["Team A", "Team B", "Other"]
    name_of = {c: names[rank] for rank, c in enumerate(order)}
    team_of = {key: name_of[int(labels[i])] for i, key in enumerate(keys)}
    swatches = {name_of[c]: _lab_to_hex(centers[c]) for c in order}
    return team_of, swatches


# =====================================================
# RANKING
# =====================================================
def rank_candidates(df, weights):
    """Score candidates against the weights. Returns (ranked table, adjusted percentiles).

    Each metric becomes a percentile among the candidates (flipped for negative weights);
    the fit score (0-100) is the weighted average. Row order matches in both frames.
    """
    weights = {m: w for m, w in weights.items() if w != 0 and m in df.columns}
    if df.empty or not weights:
        return pd.DataFrame(), pd.DataFrame()
    pool = df.reset_index(drop=True)

    pct = pd.DataFrame(index=pool.index)
    for metric, w in weights.items():
        ranked = pool[metric].rank(pct=True) * 100
        pct[metric] = ranked if w > 0 else 100 - ranked
    pct = pct.fillna(50.0)

    total = sum(abs(w) for w in weights.values())
    score = sum(pct[m] * abs(w) for m, w in weights.items()) / total

    table = pool.copy()
    table["fit_score"] = score.round(1)
    order = table["fit_score"].sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True), pct.loc[order].reset_index(drop=True)


def explain_video(row, pct_row, weights):
    """Plain-language reason for a candidate's score."""
    important = {m: w for m, w in weights.items() if w != 0 and m in pct_row.index}
    by_value = sorted(important, key=lambda m: abs(important[m]) * pct_row[m], reverse=True)
    strengths = [m for m in by_value if pct_row[m] >= 70][:3]
    by_need = sorted(important, key=lambda m: abs(important[m]), reverse=True)
    gaps = [m for m in by_need if pct_row[m] <= 35 and abs(important[m]) >= 0.5][:2]

    parts = []
    if strengths:
        parts.append("Stands out in: " + ", ".join(
            f"{VIDEO_LABELS[m].lower()} (percentile {int(round(pct_row[m]))})" for m in strengths) + ".")
    else:
        parts.append("No standout strengths against these requirements.")
    if gaps:
        parts.append("Weaker in: " + ", ".join(
            f"{VIDEO_LABELS[m].lower()} (percentile {int(round(pct_row[m]))})" for m in gaps) + ".")
    if row["observed_s"] < 5:
        parts.append(f"Only measured for {row['observed_s']:.1f} s, so treat this with caution.")
    return " ".join(parts)
