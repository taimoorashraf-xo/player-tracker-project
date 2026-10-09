import base64
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
import video_metrics as vm

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
CAMERA_MIN_RESPONSE = 0.05  # below this the pan estimate is unreliable and is ignored
CUT_PHASE_RESPONSE = 0.03   # below this the two frames share almost no picture: a new camera shot
CUT_SIMILARITY = 0.6        # colour-histogram similarity below this between frames also means a new shot
THUMB_HEIGHT = 120          # height in pixels of each player's photo

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
        "cand": None,           # candidate number shown in the video (P1, P2, ...)
        "obs": [],              # (time, foot x, foot y, box height, touches edge) in stabilised pixels
        "thumb": None,          # best photo of the player
        "thumb_h": 0,
        "color_sum": np.zeros(3),   # running sum of average shirt colour (Lab) for team grouping
        "color_n": 0,
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


def frame_signature(frame):
    """Small colour histogram of a frame, used to notice when the broadcast cuts to a new shot."""
    hsv = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1).flatten()


class CameraMotion:
    """Estimates how far a panning camera has moved, so players can be tracked in
    a stable coordinate system instead of jumping around with every pan."""

    def __init__(self):
        self.prev = None
        self.window = None
        self.offset = np.zeros(2)   # total shift of the picture content since the first frame
        self.last_response = 1.0    # how well the last two frames matched (near 0 after a cut)

    def update(self, frame):
        h, w = frame.shape[:2]
        scale = CAMERA_ANALYSIS_WIDTH / w if w > CAMERA_ANALYSIS_WIDTH else 1.0
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        if self.prev is not None and self.prev.shape == gray.shape:
            if self.window is None or self.window.shape != gray.shape:
                self.window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(self.prev, gray, self.window)
            self.last_response = response
            if response >= CAMERA_MIN_RESPONSE:
                self.offset += (dx / scale, dy / scale)
        elif self.prev is not None:
            self.last_response = 0.0
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

        label = info.get("label", str(track_id))
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
        self.shot = 0               # increases whenever the broadcast cuts to a new camera shot
        self.prev_signature = None
        self.cand_count = 0
        self.records = []           # per analysed frame: who was where (used for closeness and rendering)

    def _start_new_shot(self, frame):
        """A cut: old tracks cannot continue, so start tracking and pan estimation afresh."""
        self.shot += 1
        self.deepsort = DeepSort(
            embedder=None, max_age=30, n_init=3, nn_budget=100, max_cosine_distance=0.3,
        )
        self.camera = CameraMotion()
        self.camera.update(frame)

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

    def process_frame(self, frame, t=0.0, frame_no=0, source=None):
        """Detect and track players in one frame. Returns (annotated_frame, players).

        t is the time in seconds, frame_no the source frame number, and source an optional
        full-resolution version of the frame used for the player photos.
        """
        self.frames_processed += 1
        frame_h, frame_w = frame.shape[:2]
        photo_frame = frame if source is None else source
        photo_scale = photo_frame.shape[1] / frame_w

        # Notice cuts to a new camera shot, then track in camera-stabilised coordinates so
        # a panning camera does not make every player look like they jumped.
        signature = frame_signature(frame)
        off_x, off_y = self.camera.update(frame)
        if self.prev_signature is not None:
            similarity = cv2.compareHist(self.prev_signature, signature, cv2.HISTCMP_CORREL)
            if similarity < CUT_SIMILARITY or self.camera.last_response < CUT_PHASE_RESPONSE:
                self._start_new_shot(frame)
                off_x = off_y = 0.0
        self.prev_signature = signature

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

            key = f"S{self.shot}-{track.track_id}"
            l, top, r, b = track.to_ltrb()
            left, upper, right, lower = l + off_x, top + off_y, r + off_x, b + off_y
            edge = left < 2 or upper < 2 or right > frame_w - 2 or lower > frame_h - 2
            x1, y1 = max(0, int(left)), max(0, int(upper))
            x2, y2 = min(frame_w, int(right)), min(frame_h, int(lower))
            if x2 <= x1 or y2 <= y1:
                continue

            shirt = shirt_area(frame[y1:y2, x1:x2])
            state = self.tracks[key]
            if state["cand"] is None:
                self.cand_count += 1
                state["cand"] = self.cand_count
            state["frames"] += 1
            state["obs"].append((t, (x1 + x2) / 2 - off_x, y2 - off_y, y2 - y1, edge))

            # Average shirt colour (grass excluded), used later to group players into teams
            if shirt is not None:
                not_grass = grass_mask(cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)) == 0
                if not_grass.mean() >= 0.3:
                    lab = cv2.cvtColor(shirt, cv2.COLOR_BGR2LAB)
                    state["color_sum"] += lab[not_grass].mean(axis=0)
                    state["color_n"] += 1

            # Keep the sharpest-looking photo: the largest box that is fully in frame
            if not edge and (y2 - y1) > state["thumb_h"]:
                px1, py1 = int(x1 * photo_scale), int(y1 * photo_scale)
                px2, py2 = int(x2 * photo_scale), int(y2 * photo_scale)
                crop = photo_frame[py1:py2, px1:px2]
                if crop.size:
                    ratio = THUMB_HEIGHT / crop.shape[0]
                    state["thumb"] = cv2.resize(crop, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
                                                if ratio < 1 else cv2.INTER_CUBIC)
                    state["thumb_h"] = y2 - y1

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

            players[key] = {
                "bbox": (x1, y1, x2, y2),
                "label": f"P{state['cand']}",
                "team": _majority(state["teams"]) or "Unknown",
                "shirt_number": _majority(state["numbers"]) or "?",
                "pos": ((x1 + x2) / 2 - off_x, y2 - off_y),
                "h": y2 - y1,
                "edge": edge,
            }

        self.records.append({"t": t, "frame_no": frame_no, "players": players})
        return draw_players(frame, players), players


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
# VIDEO ANALYSIS
# =====================================================
HIGHLIGHT_COLOR = (0, 200, 255)     # BGR gold for the best-fit players


def fit_frame(frame, max_width):
    """Scale a frame down to max_width and trim to even dimensions (H.264 needs them)."""
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        h, w = frame.shape[:2]
    return frame[:h - h % 2, :w - w % 2]


def open_video_writer(path, fps):
    """H.264 writer that browsers can play. Returns None if the encoder is unavailable."""
    try:
        import imageio
        return imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)
    except Exception as e:
        st.warning(f"Could not create the output video ({e}).")
        return None


def to_data_uri(image):
    ok, buf = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def build_candidates(tracker):
    """Measurements, team and photo for every player who was visible long enough."""
    colors = {k: (st_["color_sum"] / st_["color_n"], st_["color_n"])
              for k, st_ in tracker.tracks.items() if st_["color_n"] >= 3}
    team_of, swatches = vm.cluster_teams(colors)
    contact = vm.contact_metrics(tracker.records, {k: t for k, t in team_of.items() if t != "Other"})

    rows, thumbs = [], {}
    for key, state in tracker.tracks.items():
        metrics = vm.track_metrics(state["obs"])
        if metrics is None or key not in team_of:
            continue
        extra = contact.get(key, {})
        if state["thumb"] is not None:
            thumbs[key] = cv2.imencode(".png", state["thumb"])[1].tobytes()
        rows.append({
            "key": key,
            "player": f"P{state['cand']}",
            "team": team_of[key],
            "shirt": _majority(state["numbers"]) or "",
            "photo": to_data_uri(state["thumb"]) if state["thumb"] is not None else None,
            **metrics,
            "contact_share": extra.get("contact_share", np.nan),
            "press_share": extra.get("press_share", np.nan),
        })
    return {"candidates": pd.DataFrame(rows), "swatches": swatches, "thumbs": thumbs}


def analyze_video(tracker, video_file, stride, max_width, show_preview):
    """First pass: find and track the players, then measure each one. Returns the analysis."""
    tracker.reset()
    src = video_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(src)
        video_path = tmp.name

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

        frame_no = 0
        while True:
            ret, original = cap.read()
            if not ret:
                break
            frame_no += 1
            if (frame_no - 1) % stride == 0:
                frame = fit_frame(original, max_width)
                annotated, players = tracker.process_frame(
                    frame, t=(frame_no - 1) / fps, frame_no=frame_no, source=original)
                if show_preview:
                    placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
                if total_frames > 0:
                    progress_bar.progress(min(frame_no / total_frames, 1.0))
                status_text.text(f"Frame {frame_no}/{max(total_frames, frame_no)} | "
                                 f"Players on screen: {len(players)} | Camera shots: {tracker.shot + 1}")
        cap.release()

        progress_bar.progress(1.0)
        status_text.text(f"Done. Analysed {tracker.frames_processed} of {frame_no} frames "
                         f"across {tracker.shot + 1} camera shot(s).")

        analysis = build_candidates(tracker)
        analysis.update({
            "src": src, "fps": fps, "max_width": max_width,
            "records": tracker.records, "shots": tracker.shot + 1,
        })
        return analysis

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        return None

    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


def draw_highlights(frame, players, highlights):
    """Faint boxes for everyone, bold gold boxes and a tag for the best-fit players."""
    out = frame.copy()
    frame_h = frame.shape[0]
    scale = max(0.4, frame_h / 900)
    thickness = max(2, frame_h // 200)

    for key, info in players.items():
        if key not in highlights:
            x1, y1, x2, y2 = info["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (170, 170, 170), 1)

    for key, info in players.items():
        if key in highlights:
            rank, score = highlights[key]
            x1, y1, x2, y2 = info["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), HIGHLIGHT_COLOR, thickness)
            label = f"{rank}. {info['label']}  FIT {score:.0f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2)
            tag_bottom = max(th + 6, y1)
            cv2.rectangle(out, (x1, tag_bottom - th - 6), (x1 + tw + 6, tag_bottom), HIGHLIGHT_COLOR, -1)
            cv2.putText(out, label, (x1 + 3, tag_bottom - 4), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def render_highlight_video(analysis, highlights):
    """Second pass: re-encode the video with the best-fit players highlighted."""
    boxes_at = {r["frame_no"]: r["players"] for r in analysis["records"]}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(analysis["src"])
        video_path = tmp.name
    out_path = video_path + ".out.mp4"
    writer = None

    try:
        cap = cv2.VideoCapture(video_path)
        players, frame_no = {}, 0
        while True:
            ret, original = cap.read()
            if not ret:
                break
            frame_no += 1
            frame = fit_frame(original, analysis["max_width"])
            players = boxes_at.get(frame_no, players)   # between analysed frames, reuse the last boxes
            if writer is None:
                writer = open_video_writer(out_path, analysis["fps"])
                if writer is None:
                    return None
            writer.append_data(cv2.cvtColor(draw_highlights(frame, players, highlights), cv2.COLOR_BGR2RGB))
        cap.release()
        if writer is not None:
            writer.close()
            writer = None
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "rb") as f:
                return f.read()
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
    st.write(
        "Upload match highlights (for example from YouTube) and say what kind of player you need. "
        "The AI finds and tracks the players on screen, estimates how each one moves, ranks them "
        "against your requirement, and shows the best fits in the video."
    )
    with st.expander("What this can and can't tell you"):
        st.markdown(
            "- **Speed, distance and high-intensity running are estimates.** Each player's apparent height "
            "is used as the scale, after removing the camera pan. It is indicative, not tracking-data accurate.\n"
            "- **Pressing is a proxy:** time spent closing in on an opponent. The ball is not tracked.\n"
            "- **Passing, dribbling success and exact position cannot be measured from highlights.** "
            "Players are ranked on movement and pressing only.\n"
            "- A player who appears in several camera shots shows up once per shot, because players are not "
            "recognised across cuts yet.\n"
            "- Shirt numbers are only readable in close-ups."
        )

    video_file = st.file_uploader("Upload highlights video", type=["mp4", "avi", "mov", "mkv", "flv"])
    style = st.selectbox(
        "What kind of player do you need?", list(vm.VIDEO_STYLES),
        help="Decides which measurements count most towards the fit score.",
    )

    with st.expander("Advanced settings"):
        st.caption(f"Running on: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU - the Nano model is fastest'}")
        c1, c2, c3 = st.columns(3)
        with c1:
            model_label = st.selectbox("Detection model", list(MODEL_CHOICES), index=0,
                                       help="Nano is the fastest. Medium is the most accurate but slow on CPU.")
        with c2:
            max_width = st.selectbox("Max video width (px)", WIDTH_CHOICES, index=1,
                                     help="Larger videos are scaled down to this width. Smaller is faster.")
        with c3:
            stride = st.number_input("Analyse every N frames", min_value=1, max_value=30, value=3, step=1,
                                     help="1 is the most precise and the slowest. 3 is a good balance.")
        pitch_only = st.checkbox("Only players on the pitch", value=True,
                                 help="Ignores people whose feet are not on grass. Turn off for non-grass pitches.")
        show_preview = st.checkbox("Show live preview while analysing", value=False,
                                   help="Slows processing down; the finished video plays smoothly.")

    if st.button("Analyse video", type="primary", disabled=video_file is None):
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
        st.session_state["analysis"] = analyze_video(
            tracker, video_file, int(stride), int(max_width), show_preview)
        st.session_state.pop("highlight_video", None)

    analysis = st.session_state.get("analysis")
    if analysis is not None:
        render_results(analysis, style)


def swatch_html(name, hex_color):
    box = (f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
           f"background:{hex_color};border:1px solid #888;vertical-align:middle'></span>")
    return f"{box} {name}"


def render_results(analysis, style):
    cands = analysis["candidates"]
    st.success(f"Found {len(cands)} measurable players across {analysis['shots']} camera shot(s).")
    if cands.empty:
        st.warning("No player stayed on screen long enough to measure. Try a lower 'Analyse every N frames', "
                   "a larger model, turning off 'Only players on the pitch', or a clip with longer shots.")
        return

    present = [t for t in ("Team A", "Team B", "Other") if t in set(cands["team"])]
    st.markdown("**Teams found (grouped by shirt colour):** " + "  &nbsp;&nbsp;  ".join(
        swatch_html(t, analysis["swatches"].get(t, "#888888")) for t in present), unsafe_allow_html=True)
    teams = st.multiselect(
        "Scout players from", present, default=[t for t in present if t != "Other"],
        help="'Other' is usually the referee, goalkeepers or players whose kit colour was unclear.")
    pool = cands[cands["team"].isin(teams)]
    if pool.empty:
        st.info("Select at least one team.")
        return

    base = vm.VIDEO_STYLES[style]
    weights = {}
    with st.expander("Fine-tune what matters (0 = ignore, 1 = very important)"):
        cols = st.columns(2)
        for i, metric in enumerate(vm.VIDEO_METRICS):
            with cols[i % 2]:
                weights[metric] = st.slider(
                    vm.VIDEO_LABELS[metric], 0.0, 1.0, float(base.get(metric, 0.0)), 0.1,
                    key=f"vw_{style}_{metric}", help=vm.VIDEO_HELP[metric])

    table, pct = vm.rank_candidates(pool, weights)
    if table.empty:
        st.warning("Give at least one measurement a weight above zero.")
        return
    if len(table) < 5:
        st.warning(f"Only {len(table)} players are being compared, so the percentiles are rough.")

    st.markdown(f"### Best fits for: {style.lower()}")
    top_n = st.slider("Highlight the top N players in the video", 1, min(10, len(table)), min(3, len(table)))
    shown = table.assign(rank=range(1, len(table) + 1))[[
        "rank", "photo", "player", "team", "shirt", "observed_s", "avg_speed_kmh", "top_speed_kmh",
        "distance_m", "hi_share", "press_share", "fit_score"]]
    st.dataframe(
        shown, width="stretch", hide_index=True, row_height=64,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "photo": st.column_config.ImageColumn("Photo", width="small"),
            "player": "Player", "team": "Team", "shirt": "Shirt #",
            "observed_s": st.column_config.NumberColumn("Measured (s)", format="%.1f"),
            "avg_speed_kmh": st.column_config.NumberColumn("Avg speed (km/h)", format="%.1f"),
            "top_speed_kmh": st.column_config.NumberColumn("Top speed (km/h)", format="%.1f"),
            "distance_m": st.column_config.NumberColumn("Distance (m)", format="%.0f"),
            "hi_share": st.column_config.NumberColumn("High-intensity (%)", format="%.0f"),
            "press_share": st.column_config.NumberColumn("Pressing (%)", format="%.0f"),
            "fit_score": st.column_config.ProgressColumn("Fit score", min_value=0, max_value=100, format="%.0f"),
        })
    export = shown.drop(columns=["photo"]).rename(columns={
        "rank": "Rank", "player": "Player", "team": "Team", "shirt": "Shirt", "observed_s": "Measured_s",
        "avg_speed_kmh": "Avg_speed_kmh", "top_speed_kmh": "Top_speed_kmh", "distance_m": "Distance_m",
        "hi_share": "High_intensity_pct", "press_share": "Pressing_pct", "fit_score": "Fit_score"})
    st.download_button("📥 Download ranking (CSV)", export.to_csv(index=False), "video_ranking.csv", "text/csv")

    # ---------- highlighted video ----------
    st.markdown("### Highlighted video")
    highlights = {table.loc[i, "key"]: (i + 1, float(table.loc[i, "fit_score"])) for i in range(top_n)}
    signature = tuple(sorted(highlights.items()))
    stale = st.session_state.get("highlight_signature") != signature
    if stale and (st.session_state.get("highlight_video") is None or
                  st.button("Update highlighted video for these settings")):
        with st.spinner("Rendering the highlighted video..."):
            st.session_state["highlight_video"] = render_highlight_video(analysis, highlights)
            st.session_state["highlight_signature"] = signature
    elif stale:
        st.caption("Your settings changed. Click the button above to refresh the video.")
    if st.session_state.get("highlight_video"):
        st.video(st.session_state["highlight_video"])
        st.download_button("📥 Download highlighted video", st.session_state["highlight_video"],
                           "highlighted_players.mp4", "video/mp4")
        st.caption("Gold boxes are the best fits, with their rank and fit score. Grey boxes are other players.")

    # ---------- player report ----------
    st.markdown("### Player report")
    labels = {f"{i + 1}. {table.loc[i, 'player']} ({table.loc[i, 'team']})": i for i in range(len(table))}
    chosen = st.selectbox("Player", list(labels))
    i = labels[chosen]
    row = table.loc[i]
    left, right = st.columns([1, 4])
    with left:
        if row["key"] in analysis["thumbs"]:
            st.image(analysis["thumbs"][row["key"]], width=120)
    with right:
        st.markdown(f"**{row['player']}** ({row['team']}): fit score **{row['fit_score']}** out of 100")
        st.write(vm.explain_video(row, pct.loc[i], weights))
        st.caption("Percentiles compare this player with the other players in this video only.")
    detail = pd.DataFrame({
        "Measurement": [vm.VIDEO_LABELS[m] for m in pct.columns],
        "Importance": [weights[m] for m in pct.columns],
        "Percentile (100 = best in this video)": pct.loc[i].round(0).astype(int).to_numpy(),
        "Value": [round(float(row[m]), 1) if pd.notna(row[m]) else None for m in pct.columns],
    })
    st.bar_chart(detail.set_index("Measurement")["Percentile (100 = best in this video)"])
    st.dataframe(detail, width="stretch", hide_index=True)


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

    video_tab, scouting_tab = st.tabs(["🎥 Analyse a video", "📊 Data scouting"])
    with video_tab:
        render_video_tab()
    with scouting_tab:
        render_scouting_tab()


if __name__ == "__main__":
    main()
