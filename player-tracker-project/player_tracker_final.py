"""Player Scout - Streamlit front end.

Two ways in: measure how players move in a video clip, or rank players on real match event
data. The measuring itself lives in `analysis`, `tracking` and `video_metrics`; this file
only collects the settings and draws the results.
"""
import base64
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

import analysis as an
import demo_data
import scouting
import team_context as tc
import ui_theme as ui
import video_metrics as vm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

MODEL_CHOICES = {
    "Nano - fastest": "yolov8n.pt",
    "Small - balanced": "yolov8s.pt",
    "Medium - most accurate": "yolov8m.pt",
}
WIDTH_CHOICES = [640, 854, 1280]

STYLE_NOTES = {
    "Pace and penetration": "Top speed and repeated sprints.",
    "High press": "Ground covered and time spent closing opponents down.",
    "Engine (box to box)": "Distance and sustained running.",
    "Explosive (short bursts)": "Sharp changes of pace rather than flat-out speed.",
    "All-round athlete": "An even weighting across every measurement.",
}
TEAM_DRAW_COLOURS = {"Team A": (235, 235, 235), "Team B": (90, 90, 235), "Other": (150, 150, 150)}


# =====================================================
# SHARED HELPERS
# =====================================================
@st.cache_resource(show_spinner=False)
def load_yolo(name):
    """Load a detector; missing weights download next to this script on first use."""
    return YOLO(os.path.join(BASE_DIR, name))


@st.cache_resource(show_spinner=False)
def load_ocr():
    """(reader, warning). The reader is None when EasyOCR will not load."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False), None
    except Exception as e:
        return None, f"Shirt numbers are off: EasyOCR would not load ({e})."


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_competitions():
    return scouting.list_competitions()


def to_data_uri(image):
    """BGR image -> inline PNG for the HTML player rows."""
    if image is None:
        return None
    ok, buf = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def number(value, fmt="{:.1f}", fallback="--"):
    return fmt.format(value) if isinstance(value, (int, float, np.floating)) and pd.notna(value) else fallback


# =====================================================
# VIDEO TAB
# =====================================================
def render_video_tab():
    st.caption("Upload a clip, say what kind of player you need, and the app measures how "
               "everyone on the pitch moves and ranks them.")

    with st.container(border=True):
        left, middle, right = st.columns([3, 2, 1], gap="medium", vertical_alignment="bottom")
        with left:
            clip = st.file_uploader("Clip", type=["mp4", "avi", "mov", "mkv", "flv"],
                                    label_visibility="collapsed")
        with middle:
            style = st.selectbox("Looking for", list(vm.VIDEO_STYLES), label_visibility="collapsed")
            st.caption(STYLE_NOTES.get(style, ""))
        with right:
            go = st.button("Analyse", type="primary", disabled=clip is None, width="stretch")

        with st.expander("Settings"):
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            a, b, c = st.columns(3)
            with a:
                model_label = st.selectbox("Detector", list(MODEL_CHOICES), index=0,
                                           help=f"Running on {device}. Nano is the quickest.")
            with b:
                max_width = st.selectbox("Working width", WIDTH_CHOICES, index=0,
                                         help="The clip is scaled to this before analysis.")
            with c:
                stride = st.number_input("Analyse 1 frame in", 1, 10, 2, 1,
                                         help="Lower is more accurate and slower. "
                                              "Above 3 the speeds get unreliable.")
            pitch_only = st.checkbox("Only people standing on grass", True,
                                     help="Drops the crowd and the dugout. Turn off for indoor pitches.")
            preview = st.checkbox("Show frames while analysing", False,
                                  help="Slower, but useful to confirm the detector is working.")

    if go:
        run_analysis(clip, MODEL_CHOICES[model_label], int(stride), int(max_width), pitch_only, preview)

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.caption("Clips of 20-60 seconds that follow play work best. Fast cuts and close-ups "
                   "give the tracker little to work with.")
        return
    show_results(analysis, style)


def run_analysis(clip, model_name, stride, max_width, pitch_only, preview):
    try:
        with st.spinner("Loading the detector..."):
            model = load_yolo(model_name)
            ocr, ocr_warning = load_ocr()
    except Exception as e:
        st.error(f"The detector would not load: {e}")
        return
    if ocr_warning:
        st.caption(ocr_warning)

    tracker = an.PlayerTracker(model, ocr, pitch_only=pitch_only)
    bar = st.progress(0.0, text="Starting...")
    canvas = st.empty()

    def on_progress(done, frame_no, total, count):
        bar.progress(done, text=f"frame {frame_no} of {total}  ·  {count} players in shot")

    def on_frame(frame, players):
        if preview:
            canvas.image(cv2.cvtColor(an.draw_frame(frame, players), cv2.COLOR_BGR2RGB),
                         width="stretch")

    try:
        st.session_state["analysis"] = an.analyse_clip(
            tracker, clip.read(), stride, max_width,
            on_progress=on_progress, on_frame=on_frame if preview else None)
        st.session_state.pop("highlight_video", None)
        st.session_state.pop("highlight_key", None)
    except Exception as e:
        st.error(f"Could not analyse that clip: {e}")
    bar.empty()
    canvas.empty()


def show_results(analysis, style):
    players = analysis["players"]
    if players.empty:
        st.warning("Nobody stayed on screen long enough to measure. Try analysing every frame, "
                   "a bigger detector, or a clip with longer shots.")
        return

    duration = analysis["frames"] / analysis["fps"]
    ui.render(ui.kpis([
        (len(players), "", "players measured"),
        (analysis["shots"], "", "camera shot" + ("s" if analysis["shots"] != 1 else "")),
        (f"{duration:.0f}", "s", "clip length"),
        (f"{players['top_speed_kmh'].max():.0f}", "km/h", "fastest seen"),
        (f"{players['distance_m'].sum():.0f}", "m", "ground covered"),
    ]))
    quality_note(analysis)

    # ---------- who is in the running ----------
    present = [t for t in ("Team A", "Team B", "Other") if t in set(players["team"])]
    with st.container(border=True):
        left, right = st.columns([2, 1], gap="large")
        with left:
            teams = st.multiselect("Teams to scout", present,
                                   default=[t for t in present if t != "Other"],
                                   help="Sides are grouped by kit colour. 'Other' is usually "
                                        "the referee or a goalkeeper.")
        with right:
            demo = st.checkbox("Demo mode", key="demo_mode",
                               help="Fills in made-up names and positions so the flow can be "
                                    "shown end to end. They never affect the scores.")
        named = apply_names(analysis, players, demo)
        with st.expander("Names and shirt numbers"):
            if demo:
                st.caption("Demo mode is on. Every name below is invented, not read from the "
                           "video. Say so if you present these results.")
            edited = st.data_editor(
                named[["player", "name", "shirt"]].rename(
                    columns={"player": "Track", "name": "Name", "shirt": "Shirt"}),
                hide_index=True, width="stretch", disabled=["Track"], key=f"names_{demo}")
            store_names(analysis, named, edited)
            named = apply_names(analysis, players, demo)

    pool = named[named["team"].isin(teams)] if teams else named.iloc[0:0]
    if pool.empty:
        st.info("Pick at least one team.")
        return

    weights = dict(vm.VIDEO_STYLES[style])
    with st.expander("Weight the measurements yourself"):
        columns = st.columns(2)
        for i, metric in enumerate(vm.VIDEO_METRICS):
            with columns[i % 2]:
                weights[metric] = st.slider(vm.VIDEO_LABELS[metric], 0.0, 1.0,
                                            float(weights.get(metric, 0.0)), 0.1,
                                            key=f"w_{style}_{metric}", help=vm.VIDEO_HELP[metric])

    table, pct = vm.rank_candidates(pool, weights)
    if table.empty:
        st.warning("Give at least one measurement a weight above zero.")
        return
    if len(table) < 4:
        st.caption(f"Only {len(table)} players are being compared, so the percentiles are rough.")

    # ---------- ranking ----------
    ui.render(ui.section(f"Ranked for {style.lower()}"))
    options = list(table["display"])
    if st.session_state.get("focus_player") not in options:
        st.session_state["focus_player"] = options[0]
    chosen = st.session_state["focus_player"]
    ui.render(ui.ranked_rows([{
        "rank": i + 1,
        "name": row["display"],
        "meta": f"{row['team']}  ·  {row['observed_s']:.0f}s  ·  "
                f"{row['top_speed_kmh']:.0f} km/h  ·  {row['distance_m']:.0f} m",
        "kit": analysis["swatches"].get(row["team"]),
        "photo": to_data_uri(row["thumb"]),
        "score": float(row["fit_score"]),
        "picked": row["display"] == chosen,
    } for i, row in table.iterrows()]))

    # ---------- the chosen player ----------
    ui.render(ui.section("Player report"))
    chosen = st.selectbox("Player", options, key="focus_player")
    index = int(table.index[table["display"] == chosen][0])
    row = table.loc[index]
    path = analysis["paths"].get(row["key"])

    with st.container(border=True):
        meta = " · ".join(x for x in (row["team"], f"#{row['shirt']}" if row["shirt"] else "",
                                      f"{row['observed_s']:.0f}s measured",
                                      f"{row['tracks']} spells on screen") if x)
        ui.render(ui.report_header(chosen, meta, float(row["fit_score"])))
        strengths, gaps = vm.strengths_gaps(pct.loc[index], weights)
        marks = ui.chips([f"▲ {label} p{p:.0f}" for label, p in strengths], "up") if strengths else ""
        marks += ui.chips([f"▼ {label} p{p:.0f}" for label, p in gaps], "down") if gaps else ""
        if row["observed_s"] < 5:
            marks += ui.chips([f"only {row['observed_s']:.1f}s measured"], "flat")
        if row.get("name_source") == "demo":
            marks += ui.chips(["name invented (demo)"], "flat")
        ui.render(marks or ui.chips(["nothing stands out either way"], "flat"))

        ui.render(ui.kpis([
            (f"{row['top_speed_kmh']:.1f}", "km/h", "top speed"),
            (f"{row['avg_speed_kmh']:.1f}", "km/h", "average"),
            (f"{row['distance_m']:.0f}", "m", "distance"),
            (f"{row['distance_m_per_min']:.0f}", "m/min", "work rate"),
            (f"{row['hi_share']:.0f}", "%", "high intensity"),
            (number(row["press_share"], "{:.0f}"), "%", "pressing"),
        ]))

        left, right = st.columns([5, 6], gap="medium")
        with left:
            ui.render(ui.movement_map(analysis["paths"], focus=row["key"]))
        with right:
            ui.render(ui.speed_profile(path, vm.SPRINT_MPS * 3.6, vm.HI_SPEED_MPS * 3.6))
            ui.render(ui.measure_bars([
                (vm.VIDEO_LABELS[m], vm.VIDEO_NOTES[m], float(pct.loc[index, m]),
                 f"{number(row[m])}{vm.VIDEO_UNITS[m]}") for m in pct.columns]))
            st.caption("Percentiles compare this player only with the others in this clip.")

    with st.expander("Full table and export"):
        columns = ["display", "team", "shirt", "observed_s", "tracks", "fit_score",
                   "top_speed_kmh", "avg_speed_kmh", "distance_m", "distance_m_per_min",
                   "hi_share", "sprints_per_min", "accels_per_min", "press_share"]
        export = table[columns].rename(columns={
            "display": "Player", "team": "Team", "shirt": "Shirt", "observed_s": "Measured_s",
            "tracks": "Spells", "fit_score": "Fit", "top_speed_kmh": "Top_kmh",
            "avg_speed_kmh": "Avg_kmh", "distance_m": "Distance_m",
            "distance_m_per_min": "Metres_per_min", "hi_share": "High_intensity_pct",
            "sprints_per_min": "Sprints_per_min", "accels_per_min": "Accels_per_min",
            "press_share": "Pressing_pct"})
        st.dataframe(export.round(1), width="stretch", hide_index=True)
        st.download_button("Download CSV", export.to_csv(index=False), "player_movement.csv",
                           "text/csv")

    render_clip(analysis, table)


def quality_note(analysis):
    """Say plainly when the footage made the measurements shakier than usual."""
    problems = []
    if analysis["untrusted_share"] > 0.25:
        problems.append("the camera move could not be followed in "
                        f"{analysis['untrusted_share']*100:.0f}% of frames")
    if analysis["shots"] > 1 and analysis["frames"] / analysis["fps"] / analysis["shots"] < 4:
        problems.append("the shots are very short")
    unfitted = [s for s, m in analysis["scales"].items() if not m.fitted]
    if unfitted:
        problems.append(f"{len(unfitted)} shot(s) had too few players to judge perspective, "
                        "so distances there are rough")
    heights = [1 / m.b for m in analysis["scales"].values() if m.b > 1e-6]
    if heights and not 6 <= np.median(heights) <= 45:
        problems.append(f"the implied camera height ({np.median(heights):.0f} m) is not like a "
                        "main broadcast camera, so the scale may be off")
    if problems:
        ui.render(ui.caveat("Treat the numbers below with extra caution: " + "; ".join(problems) + "."))


def apply_names(analysis, players, demo):
    """Attach a display name to every player: yours, a demo one, or the track label."""
    manual = st.session_state.setdefault("names", {})
    seed = str(len(analysis["video"]))
    rows = []
    for _, row in players.iterrows():
        invented = demo_data.simulate_player(seed + row["key"]) if demo else {}
        typed = manual.get(row["key"], {})
        name = typed.get("name") or invented.get("name") or ""
        shirt = typed.get("shirt") or row["shirt"] or invented.get("shirt") or ""
        source = "user" if typed.get("name") else ("demo" if invented.get("name") else "")
        display = name or row["player"]
        rows.append({"name": name, "shirt": shirt, "name_source": source,
                     "display": f"{display} (demo)" if source == "demo" else display})
    # The read shirt number is replaced by the resolved one, so there is only ever one
    base = players.reset_index(drop=True).drop(columns=["shirt"])
    return pd.concat([base, pd.DataFrame(rows)], axis=1)


def store_names(analysis, named, edited):
    """Keep whatever the user typed, ignoring anything that merely echoes a demo name."""
    manual = st.session_state.setdefault("names", {})
    for i in range(len(edited)):
        key = named.iloc[i]["key"]
        for field, column in (("name", "Name"), ("shirt", "Shirt")):
            typed = (edited.iloc[i][column] or "").strip()
            if typed and typed != (named.iloc[i][field] or ""):
                manual.setdefault(key, {})[field] = typed
            elif not typed and key in manual:
                manual[key].pop(field, None)


def render_clip(analysis, table):
    """The clip again, with the top players boxed."""
    ui.render(ui.section("Clip with the shortlist marked"))
    left, right = st.columns([3, 1], gap="large")
    with right:
        top_n = st.slider("Mark top", 1, min(8, len(table)), min(3, len(table))) \
            if len(table) > 1 else 1
        st.caption("Amber boxes are the shortlist, in rank order, with the fit score. "
                   "Thin boxes are everyone else, coloured by team.")

    chosen = table.head(top_n)
    highlights = {}
    for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
        for key in row["track_keys"]:
            highlights[key] = (rank, float(row["fit_score"]))
    signature = tuple(sorted((k, v) for k, v in highlights.items()))

    with left:
        if st.session_state.get("highlight_key") != signature:
            if st.button("Render clip", width="stretch"):
                with st.spinner("Rendering..."):
                    team_colours = {k: TEAM_DRAW_COLOURS.get(v) for k, v in analysis["team_of"].items()}
                    st.session_state["highlight_video"] = an.render_highlight_video(
                        analysis, highlights, team_colours)
                    st.session_state["highlight_key"] = signature
                    st.rerun()
            else:
                st.caption("The shortlist changed. Render the clip to see it marked up.")
        if st.session_state.get("highlight_video"):
            st.video(st.session_state["highlight_video"])
    with right:
        if st.session_state.get("highlight_video"):
            st.download_button("Download clip", st.session_state["highlight_video"],
                               "shortlist.mp4", "video/mp4", width="stretch")


# =====================================================
# DATA SCOUTING TAB
# =====================================================
def team_card(teams):
    """Pick a team from the data, or describe your own. Returns the team context."""
    with st.container(border=True):
        mode = st.radio("Team", ["A team in the data", "Describe my own"], horizontal=True,
                        label_visibility="collapsed", key="team_mode")
        if mode == "A team in the data" and not teams.empty:
            name = st.selectbox("Team", sorted(teams.index), key="team_name")
            row = teams.loc[name]
            formation = row["main_formation"] or "4-3-3"
            scores = tc.style_scores_from_team(row, teams)
            ui.render(ui.kpis([
                (formation, "", "usual shape"),
                (int(row["matches"]), "", "matches"),
                (f"{row['possession_pct']:.0f}", "%", "possession"),
                (f"{row['ppda']:.0f}", "", "passes per tackle (lower presses more)"),
            ]))
            st.caption(f"Plays like: {tc.style_label(scores)}.")
            if row["matches"] < 3:
                ui.render(ui.caveat(f"Only {int(row['matches'])} match(es) of data, so the "
                                    "playing style here is a rough read."))
            with st.expander("How this team compares with the rest"):
                ui.render(ui.measure_bars([
                    ("Possession", "share of all passes",
                     tc._percentile(teams["possession_pct"], row["possession_pct"]),
                     f"{row['possession_pct']:.0f}%"),
                    ("Pressing intensity", "opponent passes per defensive action",
                     tc._percentile(teams["ppda"], row["ppda"], False), f"{row['ppda']:.1f}"),
                    ("Press height", "pressures in the opponent half",
                     tc._percentile(teams["high_press_share"], row["high_press_share"]),
                     f"{row['high_press_share']:.0f}%"),
                    ("Directness", "share of passes that are long",
                     tc._percentile(teams["long_pass_share"], row["long_pass_share"]),
                     f"{row['long_pass_share']:.0f}%"),
                    ("Use of the wings", "passes and carries starting wide",
                     tc._percentile(teams["wide_share"], row["wide_share"]), f"{row['wide_share']:.0f}%"),
                    ("Defensive line", "average height of defensive actions",
                     tc._percentile(teams["def_line_x"], row["def_line_x"]), f"{row['def_line_x']:.0f}"),
                ]))
            return {"source": "data", "name": name, "formation": formation, "scores": scores,
                    "slots": {g: float(row[f"slots_{g}"]) for g in scouting.SLOT_GROUPS}}

        left, right = st.columns([1, 2], gap="large")
        with left:
            formation = st.selectbox("Shape", tc.FORMATIONS, index=1, key="custom_formation")
        with right:
            press = st.slider("Pressing", 0, 100, 50, key="custom_press")
            possession = st.slider("Possession", 0, 100, 50, key="custom_poss")
            direct = st.slider("Directness", 0, 100, 50, key="custom_direct")
        st.caption("A team described by hand has no squad to compare against, so you get the "
                   "style-aware ranking only.")
        return {"source": "custom", "name": None, "formation": formation,
                "scores": tc.style_scores_custom(press, possession, direct),
                "slots": tc.slots_for_formation(formation)}


def squad_needs(overview):
    """Where the current starters are weakest for this system."""
    used = overview[overview["in_use"]].sort_values("need", ascending=False)
    if used.empty:
        return None
    ui.render(ui.section("Squad needs"))
    ui.render(ui.measure_bars([
        (row["group"], f"{row['starter'] or 'nobody with enough minutes'} · {row['rank'] or '--'}",
         float(row["standing"]) if pd.notna(row["standing"]) else 0.0, row["verdict"])
        for _, row in used.iterrows()]))
    st.caption("How each starter ranks among everyone in the data at that position, judged on "
               "your formation and style. A shorter bar is a bigger need.")
    return used.iloc[0]["group"]


def render_scouting_tab():
    st.caption("Rank players on real match events rather than video: passing, pressing, "
               "carrying and shooting, judged against how your team plays.")
    try:
        comps = get_competitions()
    except Exception as e:
        st.error(f"Could not reach the competition list; this tab needs the internet. ({e})")
        return

    labels = comps["label"].tolist()
    with st.container(border=True):
        left, right = st.columns([3, 1], gap="medium", vertical_alignment="bottom")
        with left:
            chosen = st.multiselect("Competitions", labels,
                                    default=[l for l in labels if l == "UEFA Euro 2024"])
        with right:
            load = st.button("Load data", type="primary", disabled=not chosen, width="stretch")

    if load:
        picked = comps[comps["label"].isin(chosen)]
        bar = st.progress(0.0, text="Starting...")
        try:
            dataset = scouting.build_dataset(
                list(zip(picked["competition_id"], picked["season_id"])), DATA_CACHE_DIR,
                progress=lambda i, n: bar.progress(i / n, text=f"match {i} of {n}"))
            st.session_state.update({"profiles": dataset["profiles"], "teams": dataset["teams"],
                                     "skipped": dataset["skipped"]})
        except Exception as e:
            st.error(f"Could not download the match data: {e}")
        bar.empty()

    profiles = st.session_state.get("profiles")
    teams = st.session_state.get("teams", pd.DataFrame())
    if profiles is None or profiles.empty:
        st.caption("Pick a competition and load it. The first load downloads each match once, "
                   "then keeps it on your computer.")
        return
    ui.render(ui.kpis([(len(profiles), "", "players"), (len(teams), "", "teams"),
                       (int(profiles["matches"].max()), "", "most matches by one player")]))

    context = team_card(teams)
    need = None
    if context["source"] == "data":
        overview = tc.squad_overview(profiles, context["name"], context["scores"],
                                     context["formation"], context["slots"])
        need = squad_needs(overview)

    key = (context["source"], context["name"], context["formation"])
    st.session_state.setdefault("scout_group", "Winger")
    if st.session_state.get("scout_key") != key:
        st.session_state["scout_key"] = key
        if need:
            st.session_state["scout_group"] = need

    with st.container(border=True):
        a, b, c = st.columns(3, gap="medium")
        with a:
            group = st.selectbox("Position", scouting.POSITION_GROUPS, key="scout_group")
        with b:
            style_mode = st.selectbox("Judge them on",
                                      ["How my team plays"] + list(scouting.STYLE_WEIGHTS))
        with c:
            min_minutes = st.slider("Minimum minutes", 0, 900, 180, 30)
        versatile = st.checkbox("Include players who can also fill this position", False)

    scores = context["scores"] if style_mode == "How my team plays" else \
        ({s: 50.0 for s in tc.STYLES} if style_mode == "Balanced"
         else {s: (100.0 if s == style_mode else 0.0) for s in tc.STYLES})
    if context["slots"].get(group, 0) < 0.5:
        ui.render(ui.caveat(f"A {context['formation']} does not normally field a "
                            f"{group.lower()}. You can still scout one."))

    weights = tc.team_weights(group, scores, context["formation"], "both")
    result = tc.candidate_ranking(profiles, group, scores, context["formation"],
                                  team=context["name"], min_minutes=min_minutes,
                                  include_versatile=versatile, small_sample_adjust=True)
    if result is None or result["table"].empty:
        st.warning("Nobody matches those filters. Lower the minutes or load more matches.")
        return

    table, pct, raw, starter = result["table"], result["pct"], result["raw"], result["starter"]
    ui.render(ui.section(f"{group}s for {context['name'] or 'your team'}"))
    if starter is not None:
        st.caption(f"Compared against {starter['player']} ({int(starter['minutes'])} minutes), "
                   "your current starter there.")
    ui.render(ui.ranked_rows([{
        "rank": i + 1, "name": row["player"],
        "meta": f"{row['team']}  ·  {int(row['minutes'])} min" +
                (f"  ·  {row['vs_starter']:+.0f} vs starter" if pd.notna(row["vs_starter"]) else ""),
        "kit": None, "photo": None, "score": float(row["fit_score"]),
        "picked": i == 0,
    } for i, row in table.head(12).iterrows()]))

    ui.render(ui.section("Player report"))
    name = st.selectbox("Player", list(table["player"].head(12)))
    index = int(table.index[table["player"] == name][0])
    row = table.loc[index]
    with st.container(border=True):
        ui.render(ui.report_header(
            name, " · ".join(x for x in (row["team"], row["position"],
                                         f"{int(row['minutes'])} minutes") if x),
            float(row["fit_score"])))
        strengths, gaps = vm.strengths_gaps(pct.loc[index], weights,
                                            labels=scouting.METRIC_LABELS, min_gap_weight=0.3)
        marks = ui.chips([f"▲ {label} p{p:.0f}" for label, p in strengths], "up") if strengths else ""
        marks += ui.chips([f"▼ {label} p{p:.0f}" for label, p in gaps], "down") if gaps else ""
        if pd.notna(row["vs_starter"]):
            marks += ui.chips([f"{row['verdict']} ({row['vs_starter']:+.0f})"],
                              "up" if row["vs_starter"] >= 0 else "down")
        if row["reliability"] != "High":
            marks += ui.chips([f"{row['reliability'].lower()} sample · {int(row['minutes'])} min"], "flat")
        if row["missing_data"]:
            marks += ui.chips([f"no data for {row['missing_data']}"], "down")
        ui.render(marks)
        ui.render(ui.measure_bars([
            (scouting.METRIC_LABELS[m], scouting.METRIC_HELP[m], float(pct.loc[index, m]),
             number(raw.loc[index, m], "{:.2f}")) for m in pct.columns]))

    with st.expander("Full shortlist and export"):
        shown = table[["player", "team", "position", "minutes", "fit_score", "role_score",
                       "style_score", "reliability"]].rename(columns={
                           "player": "Player", "team": "Team", "position": "Position",
                           "minutes": "Minutes", "fit_score": "Fit", "role_score": "Role fit",
                           "style_score": "Style fit", "reliability": "Sample"})
        st.dataframe(shown.round(1), width="stretch", hide_index=True)
        st.download_button("Download CSV", shown.to_csv(index=False), "shortlist.csv", "text/csv")


# =====================================================
# APP
# =====================================================
def main():
    st.set_page_config(page_title="Player Scout", page_icon="⚽", layout="wide")
    ui.inject_css()
    ui.render(ui.masthead("Player Scout", "movement from video, fit from match data",
                          "prototype"))
    video, data = st.tabs(["Clip analysis", "Data scouting"])
    with video:
        render_video_tab()
    with data:
        render_scouting_tab()


if __name__ == "__main__":
    main()
