import os
import tempfile
from collections import Counter, defaultdict

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

import scouting

# =====================================================
# SETTINGS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Smaller models are much faster on CPU; the weights download on first use.
MODEL_CHOICES = {
    "Nano - fastest": "yolov8n.pt",
    "Small - balanced": "yolov8s.pt",
    "Medium - most accurate": "yolov8m.pt",
}
WIDTH_CHOICES = [480, 640, 960, 1280]   # frames wider than this are scaled down

PERSON_CONF = 0.4           # YOLO confidence threshold for "person"
MIN_BOX_HEIGHT_FRAC = 0.04  # ignore detections shorter than 4% of the frame height
MIN_TRACK_FRAMES = 5        # tracks seen in fewer analysed frames are dropped from the stats
OCR_EVERY_N_FRAMES = 10     # try reading a shirt number on every Nth frame per track
OCR_MAX_ATTEMPTS = 10       # give up on a track after this many OCR attempts
OCR_MIN_CONF = 0.3

# Grass in HSV (OpenCV hue is 0-180). Used to keep only people standing on the pitch
# and to ignore grass pixels when reading shirt colours.
GRASS_LOWER = (35, 40, 30)
GRASS_UPPER = (90, 255, 255)
PITCH_MIN_GRASS = 0.3       # share of grass around a person's feet to count as "on the pitch"

# Camera pan estimation
CAMERA_ANALYSIS_WIDTH = 320
CAMERA_MIN_RESPONSE = 0.05  # below this the frames are too different (a cut); ignore the shift

# Kit colours in HSV. Red wraps around, so it has two ranges. Green is left out on
# purpose because grass would otherwise be read as a green team.
TEAM_RANGES = {
    "Red": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (180, 255, 255))],
    "Blue": [((100, 100, 50), (130, 255, 255))],
    "Yellow": [((18, 100, 100), (35, 255, 255))],
    "White": [((0, 0, 170), (180, 50, 255))],
    "Black": [((0, 0, 0), (180, 255, 60))],
}
TEAM_MIN_SHARE = 0.3        # a kit colour must cover 30% of the non-grass shirt area
TEAM_BOX_COLORS = {         # BGR colours used for drawing boxes
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White": (235, 235, 235),
    "Black": (70, 70, 70),
    "Unknown": (0, 165, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _majority(counter):
    """Most common key in a Counter, or None if empty."""
    return counter.most_common(1)[0][0] if counter else None


def _new_track_state():
    return {
        "frames": 0,
        "teams": Counter(),
        "numbers": Counter(),
        "ocr_attempts": 0,
    }


def shirt_area(crop):
    """Torso region of a player crop, where the shirt colour and number are."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    torso = crop[int(0.15 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)]
    return torso if torso.size else None


def grass_mask(hsv):
    return cv2.inRange(hsv, np.array(GRASS_LOWER, np.uint8), np.array(GRASS_UPPER, np.uint8))


def on_pitch(grass, box):
    """True if the ground around the person's feet is mostly grass."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    top = max(0, int(y2 - 0.15 * h))
    bottom = min(grass.shape[0], int(y2 + 0.1 * h) + 1)
    band = grass[top:bottom, x1:x2]
    return band.size > 0 and np.count_nonzero(band) / band.size >= PITCH_MIN_GRASS


def appearance_embedding(shirt):
    """Small colour histogram of the shirt, used by the tracker to tell players apart."""
    size = 12 * 4
    if shirt is None:
        return np.full(size, 1.0 / np.sqrt(size), np.float32)
    hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    if norm == 0:
        return np.full(size, 1.0 / np.sqrt(size), np.float32)
    return (hist / norm).astype(np.float32)


class CameraMotion:
    """Estimates how far a panning camera has moved, so players can be tracked in
    a stable coordinate system instead of jumping around with every pan."""

    def __init__(self):
        self.prev = None
        self.window = None
        self.offset = np.zeros(2)   # total shift of the picture content since the first frame

    def update(self, frame):
        h, w = frame.shape[:2]
        scale = CAMERA_ANALYSIS_WIDTH / w if w > CAMERA_ANALYSIS_WIDTH else 1.0
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        if self.prev is not None and self.prev.shape == gray.shape:
            if self.window is None or self.window.shape != gray.shape:
                self.window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(self.prev, gray, self.window)
            if response >= CAMERA_MIN_RESPONSE:
                self.offset += (dx / scale, dy / scale)
        self.prev = gray
        return self.offset


def draw_players(frame, players):
    """Return a copy of frame with a thin box and a small ID tag for each player."""
    annotated = frame.copy()
    frame_h = frame.shape[0]
    thickness = 1 if frame_h < 600 else 2
    scale = max(0.35, frame_h / 1200)

    for track_id, info in players.items():
        x1, y1, x2, y2 = info["bbox"]
        color = TEAM_BOX_COLORS.get(info["team"], TEAM_BOX_COLORS["Unknown"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = str(track_id)
        if info["shirt_number"] != "?":
            label += f" #{info['shirt_number']}"
        (text_w, text_h), _ = cv2.getTextSize(label, FONT, scale, 1)
        tag_bottom = max(text_h + 4, y1)
        cv2.rectangle(annotated, (x1, tag_bottom - text_h - 4), (x1 + text_w + 4, tag_bottom), color, -1)
        text_color = (0, 0, 0) if sum(color) > 300 else (255, 255, 255)
        cv2.putText(annotated, label, (x1 + 2, tag_bottom - 3), FONT, scale, text_color, 1, cv2.LINE_AA)

    return annotated


# =====================================================
# PLAYER TRACKER CLASS
# =====================================================
class PlayerTracker:

    def __init__(self, model, ocr, pitch_only=True):
        self.model = model
        self.ocr = ocr              # may be None, in which case shirt numbers are skipped
        self.pitch_only = pitch_only
        self.reset()

    def reset(self):
        """Clear all tracking state so a new video starts from scratch."""
        # Appearance features are supplied by us (shirt colour histograms), so no
        # image embedder is created.
        self.deepsort = DeepSort(
            embedder=None, max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.3,
        )
        self.camera = CameraMotion()
        self.tracks = defaultdict(_new_track_state)
        self.frames_processed = 0

    def detect_team(self, shirt):
        """Return the dominant kit colour in the shirt area (ignoring grass), or 'Unknown'."""
        if shirt is None:
            return "Unknown"

        hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
        grass = grass_mask(hsv)
        usable = shirt.shape[0] * shirt.shape[1] - np.count_nonzero(grass)
        if usable < 0.25 * shirt.shape[0] * shirt.shape[1]:
            return "Unknown"        # mostly grass: no reliable kit colour

        best_team, best_share = "Unknown", 0.0
        for team, ranges in TEAM_RANGES.items():
            pixels = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
                pixels += np.count_nonzero(mask)
            share = pixels / usable
            if share > best_share:
                best_team, best_share = team, share

        return best_team if best_share >= TEAM_MIN_SHARE else "Unknown"

    def read_shirt_number(self, shirt):
        """Read a 1-2 digit shirt number from the shirt area with OCR."""
        if self.ocr is None or shirt is None:
            return None

        results = self.ocr.readtext(shirt, allowlist="0123456789")
        best_text, best_conf = None, 0.0
        for _, text, conf in results:
            if text.isdigit() and len(text) <= 2 and int(text) <= 99 and conf >= OCR_MIN_CONF:
                if conf > best_conf:
                    best_text, best_conf = text, conf

        return str(int(best_text)).zfill(2) if best_text else None

    def _detect_people(self, frame):
        """YOLO detections of people on the pitch, as ints (x1, y1, x2, y2) with confidence."""
        frame_h, frame_w = frame.shape[:2]
        results = self.model(frame, classes=[0], conf=PERSON_CONF, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        grass = grass_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)) if self.pitch_only else None

        people = []
        for (x1, y1, x2, y2), conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))
            if x2 <= x1 or y2 - y1 < MIN_BOX_HEIGHT_FRAC * frame_h:
                continue
            if grass is not None and not on_pitch(grass, (x1, y1, x2, y2)):
                continue
            people.append(((x1, y1, x2, y2), float(conf)))
        return people

    def process_frame(self, frame):
        """Detect and track players in one frame. Returns (annotated_frame, players)."""
        self.frames_processed += 1
        frame_h, frame_w = frame.shape[:2]

        # Track in camera-stabilised coordinates so a panning camera does not make
        # every player look like they jumped.
        off_x, off_y = self.camera.update(frame)

        detections, embeds = [], []
        for (x1, y1, x2, y2), conf in self._detect_people(frame):
            detections.append(([x1 - off_x, y1 - off_y, x2 - x1, y2 - y1], conf, "player"))
            embeds.append(appearance_embedding(shirt_area(frame[y1:y2, x1:x2])))

        tracks = self.deepsort.update_tracks(detections, embeds=embeds)

        players = {}
        for track in tracks:
            # Skip tracks that were not matched to a detection in this frame: their box
            # is only a prediction and would show up as an empty ghost box.
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x1, y1 = max(0, int(l + off_x)), max(0, int(t + off_y))
            x2, y2 = min(frame_w, int(r + off_x)), min(frame_h, int(b + off_y))
            if x2 <= x1 or y2 <= y1:
                continue

            shirt = shirt_area(frame[y1:y2, x1:x2])
            state = self.tracks[track_id]
            state["frames"] += 1

            team = self.detect_team(shirt)
            if team != "Unknown":
                state["teams"][team] += 1

            # OCR is slow, so only try on some frames, and stop after a few attempts
            due = (state["frames"] - 1) % OCR_EVERY_N_FRAMES == 0
            if due and state["ocr_attempts"] < OCR_MAX_ATTEMPTS:
                state["ocr_attempts"] += 1
                number = self.read_shirt_number(shirt)
                if number:
                    state["numbers"][number] += 1

            players[track_id] = {
                "bbox": (x1, y1, x2, y2),
                "team": _majority(state["teams"]) or "Unknown",
                "shirt_number": _majority(state["numbers"]) or "?",
            }

        return draw_players(frame, players), players

    def get_stats(self, seconds_per_frame):
        """One row per tracked player (brief, noisy tracks are dropped), longest first."""
        rows = []
        for track_id, state in self.tracks.items():
            if state["frames"] < MIN_TRACK_FRAMES:
                continue
            rows.append({
                "Player ID": track_id,
                "Shirt Number": _majority(state["numbers"]) or "N/A",
                "Team": _majority(state["teams"]) or "Unknown",
                "Time on screen (s)": round(state["frames"] * seconds_per_frame, 1),
                "Frames Analysed": state["frames"],
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Time on screen (s)", ascending=False)


# =====================================================
# CACHED MODEL LOADERS
# =====================================================
@st.cache_resource(show_spinner=False)
def load_yolo(model_name):
    # Missing weights are downloaded next to this script on first use
    return YOLO(os.path.join(BASE_DIR, model_name))


@st.cache_resource(show_spinner=False)
def load_ocr():
    """Returns (reader, error_message). The reader is None if EasyOCR failed to load."""
    try:
        return easyocr.Reader(["en"], gpu=False), None
    except Exception as e:
        return None, f"Shirt-number OCR is disabled because EasyOCR failed to load: {e}"


# =====================================================
# VIDEO PROCESSING
# =====================================================
def open_video_writer(path, fps):
    """H.264 writer that browsers can play. Returns None if the encoder is unavailable."""
    try:
        import imageio
        return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)
    except Exception as e:
        st.warning(f"Could not create the output video ({e}). Statistics will still be produced.")
        return None


def process_video(tracker, video_file, stride, max_width, show_preview):
    """Run the tracker over an uploaded video.

    Returns {"stats": DataFrame, "video": mp4 bytes or None}, or None on failure.
    Every source frame is written to the output video; frames that are skipped
    for speed reuse the most recent boxes so the result plays back smoothly.
    """
    tracker.reset()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name
    out_path = video_path + ".out.mp4"
    writer = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30
        st.info(f"📊 Frames: {total_frames} | FPS: {fps:.1f} | Analysing every {stride} frame(s)")

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        placeholder = st.empty()

        players = {}
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
            h, w = frame.shape[:2]
            frame = frame[:h - h % 2, :w - w % 2]    # H.264 needs even dimensions

            if (frame_no - 1) % stride == 0:
                annotated, players = tracker.process_frame(frame)
                if show_preview:
                    placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
            else:
                annotated = draw_players(frame, players)

            if writer is None and frame_no == 1:
                writer = open_video_writer(out_path, fps)
            if writer is not None:
                writer.append_data(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            if frame_no % 5 == 0 or frame_no == total_frames:
                if total_frames > 0:
                    progress_bar.progress(min(frame_no / total_frames, 1.0))
                    status_text.text(f"Frame {frame_no}/{total_frames} | Players on screen: {len(players)}")
                else:
                    status_text.text(f"Frame {frame_no} | Players on screen: {len(players)}")

        cap.release()
        if writer is not None:
            writer.close()
            writer = None

        progress_bar.progress(1.0)
        status_text.text(f"Done. Analysed {tracker.frames_processed} of {frame_no} frames.")

        video_bytes = None
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                video_bytes = f.read()

        return {"stats": tracker.get_stats(stride / fps), "video": video_bytes}

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        return None

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        for path in (video_path, out_path):
            try:
                os.unlink(path)
            except OSError:
                pass


# =====================================================
# VIDEO TAB
# =====================================================
def render_video_tab():
    st.caption(f"Running on: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU - use the Nano model for speed'}")
    st.info(
        "This tab tracks players in video you upload and shows who is on screen. It works best on a "
        "fixed or slowly moving camera. It cannot measure speed, distance or passing from edited "
        "highlights; use the Scouting tab for those."
    )

    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "flv"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        model_label = st.selectbox(
            "Detection model", list(MODEL_CHOICES), index=0,
            help="Nano is the fastest. Medium is the most accurate but slow on CPU.",
        )
    with c2:
        max_width = st.selectbox(
            "Max video width (px)", WIDTH_CHOICES, index=1,
            help="Larger videos are scaled down to this width. Smaller is faster.",
        )
    with c3:
        stride = st.number_input(
            "Analyse every N frames", min_value=1, max_value=30, value=3, step=1,
            help="1 analyses every frame (slowest, most precise). Skipped frames reuse the latest boxes.",
        )
    with c4:
        pitch_only = st.checkbox(
            "Only players on the pitch", value=True,
            help="Ignores people in the crowd or beside the pitch by checking for grass at their feet. "
                 "Turn off for non-grass or indoor pitches.",
        )
        show_preview = st.checkbox(
            "Show live preview", value=False,
            help="Shows frames while processing. This slows processing down; the finished video plays smoothly.",
        )
    with c5:
        st.write("")
        process_btn = st.button("Process", type="primary", disabled=video_file is None)

    if video_file is not None and process_btn:
        try:
            with st.spinner("Loading models (the first use of a model downloads it)..."):
                model = load_yolo(MODEL_CHOICES[model_label])
                ocr, ocr_warning = load_ocr()
        except Exception as e:
            st.error(f"❌ Failed to load models: {e}")
            st.stop()

        if ocr_warning:
            st.warning(ocr_warning)

        tracker = PlayerTracker(model, ocr, pitch_only=pitch_only)
        st.session_state["result"] = process_video(
            tracker, video_file, int(stride), int(max_width), show_preview
        )

    # Results live in session state so that clicking Download does not wipe them
    result = st.session_state.get("result")
    if result is not None:
        if result["video"]:
            st.subheader("🎬 Annotated Video")
            st.video(result["video"])
            st.download_button("📥 Download video", result["video"], "tracked_players.mp4", "video/mp4")

        st.subheader("📊 Player Statistics")
        stats_df = result["stats"]
        if stats_df.empty:
            st.warning("⚠️ No players tracked for long enough. Try turning off 'Only players on the pitch', "
                       "using a larger model, or a lower N.")
        else:
            st.dataframe(stats_df, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download CSV",
                stats_df.to_csv(index=False),
                "player_stats.csv",
                "text/csv",
            )


# =====================================================
# SCOUTING TAB
# =====================================================
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_competitions():
    return scouting.list_competitions()


def render_scouting_tab():
    st.write(
        "Describe the position and the way your team plays. The app scores every player in the "
        "dataset against that description and explains the fit. All numbers come from real match "
        "event data (free StatsBomb Open Data), not from video."
    )
    with st.expander("What this can and can't tell you"):
        st.markdown(
            "- **Passing, pressing, carrying and shooting** are counted from real match events.\n"
            "- **Work rate and pace are proxies.** Event data has no distance run or sprint speed, so "
            "work rate uses pressures, recoveries and counter-presses, and pace uses progressive carries, "
            "carry distance and dribbles.\n"
            "- Players are compared **only with others in the same position group**, as percentiles.\n"
            "- The free data covers selected competitions only, and a few matches means a small sample. "
            "Use the result to build a shortlist to watch, not as a final verdict."
        )

    # ---------- 1. data ----------
    try:
        comps = get_competitions()
    except Exception as e:
        st.error(f"Could not load the competition list. An internet connection is needed. ({e})")
        return

    labels = comps["label"].tolist()
    chosen = st.multiselect(
        "Competitions to analyse", labels,
        default=[l for l in labels if l == "UEFA Euro 2024"],
        help="Choosing several gives bigger player pools. The first load downloads match data "
             "(about a second per match) and is then cached on your computer.",
    )
    max_matches = st.number_input("Limit number of matches (0 = all)", 0, 1000, 0, step=1)

    if st.button("Load player data", disabled=not chosen):
        picked = comps[comps["label"].isin(chosen)]
        selections = list(zip(picked["competition_id"], picked["season_id"]))
        bar = st.progress(0.0, text="Starting...")
        try:
            profiles, skipped = scouting.build_dataset(
                selections, DATA_CACHE_DIR, int(max_matches) or None,
                progress=lambda i, n: bar.progress(i / n, text=f"Match {i} of {n}"),
            )
            st.session_state["profiles"] = profiles
            st.session_state["skipped"] = skipped
        except Exception as e:
            st.error(f"Could not download match data: {e}")
        bar.empty()

    profiles = st.session_state.get("profiles")
    if profiles is None:
        st.info("Choose competitions and click **Load player data** to begin.")
        return
    if profiles.empty:
        st.warning("No player data was loaded.")
        return
    note = f" ({st.session_state['skipped']} matches could not be read)" if st.session_state.get("skipped") else ""
    st.success(f"Loaded {len(profiles)} players{note}.")

    # ---------- 2. requirements ----------
    st.markdown("### Your requirements")
    c1, c2, c3 = st.columns(3)
    with c1:
        group = st.selectbox("Position you are recruiting for", scouting.POSITION_GROUPS, index=5)
    with c2:
        style = st.selectbox("How does your team play?", list(scouting.STYLE_WEIGHTS))
    with c3:
        min_minutes = st.slider("Minimum minutes played", 0, 900, 180, step=30)

    base = scouting.default_weights(group, style)
    metric_keys = list(scouting.RATE_METRICS) + list(scouting.OTHER_METRICS)
    weights = {}
    with st.expander("Fine-tune what matters (-1 = lower is better, 0 = ignore, 1 = very important)"):
        cols = st.columns(2)
        for i, metric in enumerate(metric_keys):
            with cols[i % 2]:
                weights[metric] = st.slider(
                    scouting.METRIC_LABELS[metric], -1.0, 1.0, float(base.get(metric, 0.0)), 0.1,
                    key=f"w_{group}_{style}_{metric}", help=scouting.METRIC_HELP[metric],
                )

    # ---------- 3. results ----------
    table, pct, raw = scouting.fit_scores(profiles, group, weights, min_minutes)
    if table.empty:
        st.warning("No players match these filters. Lower the minimum minutes or load more matches.")
        return
    if len(table) < 8:
        st.warning(f"Only {len(table)} players are in this comparison group, so percentiles are rough.")

    st.markdown(f"### Best fits: {group.lower()}s for a {style.lower()} team")
    top_n = st.slider("Players to show", 5, 50, min(15, len(table)))
    shown = table.head(top_n).rename(columns={
        "player": "Player", "team": "Team", "position": "Position",
        "minutes": "Minutes", "fit_score": "Fit score (0-100)"})
    st.dataframe(shown, width="stretch", hide_index=True)
    st.download_button("📥 Download shortlist (CSV)", shown.to_csv(index=False), "shortlist.csv", "text/csv")

    st.markdown("### Player report")
    name = st.selectbox("Player", shown["Player"].tolist())
    i = int(table.index[table["player"] == name][0])
    st.markdown(f"**{name}** ({table.loc[i, 'team']}): fit score **{table.loc[i, 'fit_score']}** out of 100")
    st.write(scouting.explain(table.loc[i], pct.loc[i], weights))

    detail = pd.DataFrame({
        "Metric": [scouting.METRIC_LABELS[m] for m in pct.columns],
        "Importance": [weights[m] for m in pct.columns],
        "Percentile (100 = best among peers)": pct.loc[i].round(0).astype(int).to_numpy(),
        "Value": raw.loc[i].round(2).to_numpy(),
    })
    st.bar_chart(detail.set_index("Metric")["Percentile (100 = best among peers)"])
    st.dataframe(detail, width="stretch", hide_index=True)


# =====================================================
# MAIN APP
# =====================================================
def main():
    st.set_page_config(page_title="⚽ Player Scouting", layout="wide")
    st.title("⚽ Player Scouting & Tracking")

    scouting_tab, video_tab = st.tabs(["🔎 Scouting", "🎥 Video tracker"])
    with scouting_tab:
        render_scouting_tab()
    with video_tab:
        render_video_tab()


if __name__ == "__main__":
    main()
